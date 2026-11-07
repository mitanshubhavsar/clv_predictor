"""Write everything the public site and the dashboard read, from the real saved model and data.
Nothing on the site is typed by hand: every number comes from outputs/metrics.json, the saved model, or the data.

Writes
  outputs/test_predictions.csv      one row per test customer: features, prediction, probability, segment, actual
  outputs/model_comparison.csv      5-fold CV on train for every model we tried (same settings as the steps)
  outputs/shap_importance.csv       mean |SHAP| of model C on the multiplier (train)
  site/public/data/{summary,capture_curve,dataset,scorer}.json

Run (needs outputs/ from step15 and data/ledger.csv, data/raw.csv):
    PYTHONNOUSERSITE=1 python scripts/export_site_data.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import evaluate, mae, oracle_capture, r2, revenue_capture  # noqa: E402
from step3_split import TEST_CUTOFF, TRAIN_CUTOFF, load_ledger, make_snapshot  # noqa: E402
from step4_rfm import build_rfm  # noqa: E402
from step6_baselines import BASELINES, cv_predictions  # noqa: E402
from step7a_ridge_log import LogRidge  # noqa: E402
from step7b_xgboost import XGBModel  # noqa: E402
from step7c_tweedie import XGBTweedie  # noqa: E402
from step7d_hurdle import BEST_7D_LOG, HurdleModel  # noqa: E402
from step7e_ratio import ALPHA_7A_SMEAR, BEST_7B, BEST_7C, BEST_7E_TW, RatioXGB  # noqa: E402
from step8_features import build_features  # noqa: E402
from step8_final import BEST_8, KEPT  # noqa: E402
from step10_segments import segment, summarize  # noqa: E402

OUT = ROOT / "outputs"
SITE = ROOT / "site" / "public" / "data"
FEATURES = ["Recency", "Frequency", "Monetary", "IPI", "OverdueRatio"]
SEGS = ["Protect", "Win back", "Grow", "Low priority"]

FEATURE_SCHEMA = {
    "Recency": "Days from the customer's last purchase to the cutoff date.",
    "Frequency": "Number of distinct purchase invoices in the 12-month window.",
    "Monetary": "Net spend in the window: purchases minus returns, in GBP.",
    "IPI": "Mean days between purchase days; blank if the customer bought on only one day.",
    "OverdueRatio": "Recency divided by the customer's own usual gap (at least 1 day); blank when IPI is blank.",
}
RAW_SCHEMA = {
    "Invoice": ("text", "Invoice number. A leading 'C' marks a cancellation."),
    "StockCode": ("text", "Product code."),
    "Description": ("text", "Product name."),
    "Quantity": ("integer", "Units on this line; negative on cancellations."),
    "InvoiceDate": ("datetime", "When the invoice was placed."),
    "Price": ("number", "Unit price in GBP."),
    "Customer ID": ("text", "Customer identifier. Missing on 23% of rows; those are dropped."),
    "Country": ("text", "Customer's country."),
}
PRED_SCHEMA = {
    "predicted_spend": "Model C's predicted money kept over the next 6 months, in GBP.",
    "actual_spend": "What the customer actually kept in those 6 months (floored at 0), in GBP.",
    "probability_buy_again": "Predicted chance of any net spend in the next 6 months.",
    "segment": "Value x return-probability group: Protect, Win back, Grow, Low priority.",
}


def clean_json(o):
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in (o.tolist() if isinstance(o, np.ndarray) else o)]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d %H:%M")
    return o


def write(name: str, obj) -> None:
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / name).write_text(json.dumps(clean_json(obj), separators=(",", ":"), allow_nan=False))
    print(f"wrote {name} ({(SITE / name).stat().st_size / 1024:.1f} KB)")


def capture_curve(y: np.ndarray, score: np.ndarray, grid: np.ndarray) -> list[float]:
    order = np.argsort(-score, kind="stable")
    cum = np.cumsum(y[order]) / y.sum()
    return [0.0 if g == 0 else float(cum[max(1, int(round(len(y) * g))) - 1]) for g in grid]


def main() -> None:
    metrics = json.load(open(OUT / "metrics.json"))
    spec = json.load(open(OUT / "model_spec.json"))
    mc = joblib.load(OUT / "model_c.joblib"); model, cols = mc["model"], mc["columns"]
    clf = joblib.load(OUT / "logistic_return.joblib")["model"]
    cutoff_gbp = metrics["segments_test"]["high_value_cutoff_gbp_from_train"]

    ledger = load_ledger()
    tr, te = make_snapshot(ledger, TRAIN_CUTOFF), make_snapshot(ledger, TEST_CUTOFF)
    Xtr, Xte = build_features(tr, KEPT), build_features(te, KEPT)
    y = te.target.to_numpy()

    # ---- test predictions (saved file the dashboard reads) -------------------
    pred = model.predict(Xte[cols])
    prob = clf.predict_proba(Xte)
    seg = segment(pred >= cutoff_gbp, prob > 0.5).to_numpy()
    tp = Xte[FEATURES].copy()
    tp["predicted_spend"], tp["multiplier"] = pred, pred / model.scale(Xte[cols])
    tp["probability_buy_again"], tp["segment"], tp["actual_spend"] = prob, seg, y
    tp.round(4).to_csv(OUT / "test_predictions.csv")
    print(f"wrote outputs/test_predictions.csv ({len(tp)} customers)")

    # ---- model comparison on train, 5-fold CV, same settings as steps 6 to 8 --
    Xr, ytr = build_rfm(tr), tr.target
    runs = {**{k: (c, Xr) for k, c in BASELINES.items()},
            "Log-Ridge + smearing (7a)": (lambda: LogRidge(ALPHA_7A_SMEAR, True), Xr),
            "XGBoost on raw GBP (7b)": (lambda: XGBModel(**BEST_7B), Xr),
            "XGBoost Tweedie (7c)": (lambda: XGBTweedie(**BEST_7C), Xr),
            "Hurdle, log + smearing (7d)": (lambda: HurdleModel(log_stage2=True, **BEST_7D_LOG), Xr),
            "Ratio Tweedie, R,F,M (7e)": (lambda: RatioXGB(objective="tweedie", **BEST_7E_TW), Xr),
            "Model C: ratio Tweedie + interval/overdue": (lambda: RatioXGB(objective="tweedie", **BEST_8), Xtr[cols])}
    rows = {}
    for k, (mk, X) in runs.items():
        e = evaluate(ytr.to_numpy(), cv_predictions(mk, X, ytr))
        rows[k] = {"R2": e["R2"], "MAE": e["MAE"], "Capture@20%": e["Capture@20%"]}
    mcmp = pd.DataFrame(rows).T.rename(index={"mean": "Mean baseline", "past_spend": "Past spend (0.37 x Monetary)"})
    mcmp.round(4).to_csv(OUT / "model_comparison.csv")
    print("wrote outputs/model_comparison.csv")

    # ---- SHAP on the multiplier (train) --------------------------------------
    sv = shap.TreeExplainer(model.model_)(Xtr[cols]).values
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    shp = pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.values, "share": imp.values / imp.sum()})
    shp.round(4).to_csv(OUT / "shap_importance.csv", index=False)

    # ---- curves ---------------------------------------------------------------
    grid = np.round(np.linspace(0, 1, 101), 2)
    past = Xte["Monetary"].to_numpy()
    curves = {"Model C": capture_curve(y, pred, grid), "Sort by past spend": capture_curve(y, past, grid),
              "Recency only": capture_curve(y, -Xte["Recency"].to_numpy(), grid),
              "Perfect foresight": capture_curve(y, y, grid), "Random": [float(g) for g in grid]}
    k20 = int(np.where(grid == 0.2)[0][0])
    assert abs(curves["Model C"][k20] - metrics["test"]["model_C_headline"]["Capture@20%"]["value"]) < 0.005
    write("capture_curve.json", {"grid": grid, "curves": curves})

    # ---- summary ---------------------------------------------------------------
    rng = np.random.default_rng(0); d = {"R2": [], "MAE gain": [], "Capture@20%": []}
    from step6_baselines import PastSpendBaseline
    pp = PastSpendBaseline().fit(build_rfm(tr), ytr).predict(build_rfm(te))
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        d["R2"].append(r2(y[i], pred[i]) - r2(y[i], pp[i]))
        d["MAE gain"].append(mae(y[i], pp[i]) - mae(y[i], pred[i]))
        d["Capture@20%"].append(revenue_capture(y[i], pred[i]) - revenue_capture(y[i], pp[i]))
    diff = {k: {"point": float(np.mean(v)), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in d.items()}
    raw = pd.read_csv(ROOT / "data" / "raw.csv", parse_dates=["InvoiceDate"])
    seg_tbl = summarize(pd.Series(seg), te.target).reset_index().rename(columns={"seg": "segment"})
    write("summary.json", {
        "model": spec["model"], "columns": cols, "params": spec["params"],
        "cutoff_train": spec["train_cutoff"], "cutoff_test": spec["test_cutoff"],
        "n_train": metrics["data"]["train_customers"], "n_test": metrics["data"]["test_customers"],
        "zero_share_train": metrics["data"]["train_share_zero_target"], "zero_share_test": metrics["data"]["test_share_zero_target"],
        "test": metrics["test"], "capture_ceiling": metrics["test_capture_ceiling"],
        "train_cv": metrics["train_cv_fresh_seeds_20_29"], "classifier_auc": metrics["return_classifier_train_cv_auc"],
        "diff_vs_past_spend": diff, "targeting": metrics["targeting_test_top20"],
        "high_value_cutoff_gbp": cutoff_gbp, "segments": seg_tbl.to_dict("records"),
        "model_comparison": [{"model": k, **v} for k, v in mcmp.round(4).to_dict("index").items()],
        "shap": shp.to_dict("records"),
        "raw_rows": int(len(raw)), "raw_customers": int(raw["Customer ID"].nunique()),
        "raw_first_date": raw["InvoiceDate"].min(), "raw_last_date": raw["InvoiceDate"].max(),
    })

    # ---- dataset tables: one customer followed through all three ----------------
    rng = np.random.default_rng(7)
    counts = te.feature_rows.groupby("Customer ID").size()
    cand = [c for c in Xte.index[seg == "Protect"] if 12 <= counts[c] <= 25 and 4 <= Xte.loc[c, "Frequency"] <= 8]
    followed = int(rng.choice(sorted(cand)))
    raw_cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    rr = te.feature_rows[te.feature_rows["Customer ID"] == followed].sort_values("InvoiceDate")[raw_cols]
    sample = [followed] + [int(c) for c in rng.choice([c for c in Xte.index if c != followed], 24, replace=False)]
    null_pct = (pd.read_csv(ROOT / "data" / "raw.csv", usecols=["Customer ID"]).isna().mean() * 100).round(2)
    ex = raw.dropna(subset=["Customer ID"]).iloc[0]
    sel = tp.loc[sample]
    write("dataset.json", {
        "followed_customer": str(followed),
        "raw": {"schema": [{"column": c, "type": t, "meaning": m, "example": str(ex[c]),
                            "null_pct": float(null_pct["Customer ID"]) if c == "Customer ID" else 0.0} for c, (t, m) in RAW_SCHEMA.items()],
                "columns": raw_cols, "rows": rr.astype(object).where(rr.notna(), None).values.tolist(),
                "note": f"All {len(rr)} rows of customer {followed} inside the feature window "
                        f"({te.feature_start:%Y-%m-%d} to {(TEST_CUTOFF - pd.Timedelta(days=1)):%Y-%m-%d})."},
        "features": {"schema": [{"column": c, "meaning": FEATURE_SCHEMA[c], "p5": float(Xte[c].quantile(.05)),
                                 "median": float(Xte[c].median()), "p95": float(Xte[c].quantile(.95))} for c in FEATURES],
                     "columns": ["Customer ID"] + FEATURES,
                     "rows": [[str(c)] + Xte.loc[c, FEATURES].astype(float).round(2).tolist() for c in sample]},
        "predictions": {"schema": [{"column": c, "meaning": m} for c, m in PRED_SCHEMA.items()],
                        "columns": ["Customer ID", "predicted_spend", "actual_spend", "probability_buy_again", "segment"],
                        "rows": [[str(c), round(float(sel.loc[c, "predicted_spend"]), 2), round(float(sel.loc[c, "actual_spend"]), 2),
                                  round(float(sel.loc[c, "probability_buy_again"]), 3), sel.loc[c, "segment"]] for c in sample]},
    })

    # ---- live-scorer data -------------------------------------------------------
    presets = []
    for s in SEGS:
        inseg = tp[tp["segment"] == s]["predicted_spend"]
        lo, hi = inseg.quantile([0.4, 0.6])
        ids = sorted(inseg.index[(inseg >= lo) & (inseg <= hi)])
        cid = int(rng.choice(ids)); r = tp.loc[cid]
        presets.append({"segment": s, "customer_id": str(cid),
                        "features": {k: (None if pd.isna(r[k]) else round(float(r[k]), 2)) for k in ["Recency", "Frequency", "Monetary", "IPI"]},
                        "predicted_spend": round(float(r["predicted_spend"]), 2),
                        "probability_buy_again": round(float(r["probability_buy_again"]), 4), "actual_spend": round(float(r["actual_spend"]), 2)})
    ranges = {c: {"min": float(Xte[c].min()), "p1": float(Xte[c].quantile(.01)), "median": float(Xte[c].median()),
                  "p99": float(Xte[c].quantile(.99)), "max": float(Xte[c].max())} for c in ["Recency", "Frequency", "Monetary", "IPI"]}
    write("scorer.json", {"presets": presets, "ranges": ranges, "pred_quantiles": np.quantile(pred, np.linspace(0, 1, 101)),
                          "n_test": int(len(tp)), "high_value_cutoff_gbp": cutoff_gbp,
                          "note": "Presets are seeded random draws from the middle 20% (by predicted spend) of each segment on the test snapshot, so none is an outlier."})


if __name__ == "__main__":
    main()
