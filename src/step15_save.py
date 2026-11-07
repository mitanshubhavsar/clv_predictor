"""Step 15: save the final model (C) and a metrics file. Reproduces the step 9 / 10 / 12 numbers.

Writes (under outputs/):
  model_c.joblib      fitted RatioXGB (Tweedie) on ALL train customers, plus the column list
  logistic_return.joblib  the "likely to return" classifier (logistic on R,F,M)
  model_spec.json     everything needed to rebuild the features and the segments
  metrics.json        train CV (fresh fold seeds), test metrics with 95% intervals, segments, targeting lift

Load with:  sys.path.insert(0, "src"); joblib.load("outputs/model_c.joblib")
(the classes live in src/step7e_ratio.py and src/step10_repurchase_segmentation.py, so src must be importable)

Run:  python src/step15_save.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from metrics import bootstrap_ci, drop_top, evaluate, oracle_capture
from step3_split import TEST_CUTOFF, TRAIN_CUTOFF, load_ledger, make_snapshot
from step6_baselines import MeanBaseline, PastSpendBaseline
from step7b_xgboost import oof_with_seed
from step7e_ratio import BEST_7E_TW, FLOOR, RatioXGB
from step8_features import build_features
from step8_final import BEST_8, KEPT
from step10_repurchase_segmentation import LogisticRFM, oof
from step10_segments import segment, summarize
from step12_targeting import top_mask

OUT = Path(__file__).resolve().parent.parent / "outputs"
COLUMNS = ["Recency", "Frequency", "Monetary", *KEPT]
r = lambda v, n=4: None if v is None else round(float(v), n)


def metric_block(y, p):
    e = evaluate(y, p)
    out = {m: {"value": r(e[m], 3), "ci95": [r(x, 3) for x in bootstrap_ci(y, p, m)]} for m in ("R2", "MAE", "Capture@20%")}
    y2, p2 = drop_top(y, p, 10); e2 = evaluate(y2, p2)
    out["without_top10"] = {m: r(e2[m], 3) for m in ("R2", "MAE", "Capture@20%")}
    out["predicted_total_over_actual"] = r(p.sum() / y.sum(), 3)
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ledger = load_ledger()
    tr, te = make_snapshot(ledger, TRAIN_CUTOFF), make_snapshot(ledger, TEST_CUTOFF)
    Xtr, Xte = build_features(tr, KEPT), build_features(te, KEPT)
    ytr, yte = tr.target, te.target

    model = RatioXGB(objective="tweedie", **BEST_8).fit(Xtr[COLUMNS], ytr)
    clf = LogisticRFM().fit(Xtr, (ytr > 0).astype(int))
    joblib.dump({"model": model, "columns": COLUMNS}, OUT / "model_c.joblib")
    joblib.dump({"model": clf, "columns": ["Recency", "Frequency", "Monetary"]}, OUT / "logistic_return.joblib")

    # train: honest CV on fresh fold seeds 20..29 (same protocol as step 13)
    ya = ytr.to_numpy()
    cv = [evaluate(ya, oof_with_seed(lambda: RatioXGB(objective="tweedie", **BEST_8), Xtr[COLUMNS], ytr, s)) for s in range(20, 30)]
    train_cv = {m: {"mean": r(np.mean([e[m] for e in cv]), 3), "sd": r(np.std([e[m] for e in cv]), 3)} for m in ("R2", "MAE", "Capture@20%")}

    # test
    yt = yte.to_numpy()
    pc = model.predict(Xte[COLUMNS])
    preds = {"mean": MeanBaseline().fit(build_features(tr), ytr).predict(build_features(te)),
             "past_spend": PastSpendBaseline().fit(build_features(tr), ytr).predict(build_features(te)),
             "model_A_RFM": RatioXGB(objective="tweedie", **BEST_7E_TW).fit(build_features(tr), ytr).predict(build_features(te)),
             "model_C_headline": pc}
    test = {k: metric_block(yt, p) for k, p in preds.items()}

    # segments and targeting (cut-off fixed on train out-of-fold predictions)
    cutoff = float(np.quantile(oof_with_seed(lambda: RatioXGB(objective="tweedie", **BEST_8), Xtr[COLUMNS], ytr, 0), 0.8))
    prob = clf.predict_proba(Xte)
    seg = summarize(segment(pc >= cutoff, prob > 0.5), yte)
    m = top_mask(pc); mp = top_mask(preds["past_spend"])
    auc = float(np.mean([roc_auc_score((ytr > 0).astype(int), oof(LogisticRFM, Xtr, (ytr > 0).astype(int), s)) for s in range(3)]))

    metrics = {
        "data": {"train_customers": len(ytr), "test_customers": len(yte),
                 "train_share_zero_target": r((ytr == 0).mean(), 3), "test_share_zero_target": r((yte == 0).mean(), 3)},
        "train_cv_fresh_seeds_20_29": train_cv,
        "test": test,
        "test_capture_ceiling": r(oracle_capture(yt), 3),
        "segments_test": {"high_value_cutoff_gbp_from_train": r(cutoff, 0), **json.loads(seg.reset_index().to_json(orient="records").join(['{"rows":', '}'])) },
        "targeting_test_top20": {"model_C_mean_future_spend": r(yt[m].mean(), 0), "past_spend_mean_future_spend": r(yt[mp].mean(), 0),
                                 "average_customer": r(yt.mean(), 0), "model_C_share_of_money": r(yt[m].sum() / yt.sum(), 3)},
        "return_classifier_train_cv_auc": r(auc, 3),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    spec = {"model": "RatioXGB (reg:tweedie on future/scale), prediction = multiplier x max(Monetary, %g), clipped at 0" % FLOOR,
            "params": BEST_8, "columns": COLUMNS, "train_cutoff": str(TRAIN_CUTOFF.date()), "test_cutoff": str(TEST_CUTOFF.date()),
            "window": "12 months of features before the cutoff, 6 months of target after it",
            "target": "net money kept in the 6 months after the cutoff, floored at 0 (returns netted off, dated at cancellation)",
            "feature_definitions": {"Recency": "days from last purchase to cutoff", "Frequency": "distinct purchase invoices",
                                    "Monetary": "net spend (purchases minus returns) in the feature window",
                                    "IPI": "mean days between purchase days; blank (NaN) if one purchase day",
                                    "OverdueRatio": "Recency / max(IPI, 1); blank if IPI blank"},
            "segments": {"high_value": "predicted spend >= %.0f GBP (80th pct of train out-of-fold predictions)" % cutoff,
                         "likely_to_return": "logistic P(spend > 0) > 0.5"}}
    (OUT / "model_spec.json").write_text(json.dumps(spec, indent=2))
    print(json.dumps({k: metrics[k] for k in ("data", "train_cv_fresh_seeds_20_29", "targeting_test_top20", "return_classifier_train_cv_auc")}, indent=1))
    print("test C:", metrics["test"]["model_C_headline"]["R2"], metrics["test"]["model_C_headline"]["MAE"], metrics["test"]["model_C_headline"]["Capture@20%"])


if __name__ == "__main__":
    main()
