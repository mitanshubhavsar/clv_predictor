"""Step 9: the ONE evaluation on the untouched test snapshot (cutoff 2011-06-01).

Everything is FIT on the train snapshot only, then applied to the test snapshot:
  mean baseline, past-spend baseline (k fit on train), model A (7e settings, R,F,M),
  model C (headline: re-tuned, R,F,M + IPI + OverdueRatio).
Reports R2, MAE, Capture@20% with 95% bootstrap intervals over customers, the
same without the 10 biggest customers, and C minus past-spend paired intervals.

Run:  python src/step9_test.py
"""
import numpy as np
import pandas as pd

from metrics import bootstrap_ci, drop_top, evaluate, mae, oracle_capture, r2, revenue_capture
from step3_split import TEST_CUTOFF, TRAIN_CUTOFF, load_ledger, make_snapshot
from step6_baselines import MeanBaseline, PastSpendBaseline
from step7e_ratio import BEST_7E_TW, RatioXGB
from step8_features import build_features
from step8_final import BEST_8, KEPT


def main() -> None:
    ledger = load_ledger()
    tr, te = make_snapshot(ledger, TRAIN_CUTOFF), make_snapshot(ledger, TEST_CUTOFF)
    ytr, yte = tr.target, te.target
    yt = yte.to_numpy()
    Xr_tr, Xr_te = build_features(tr), build_features(te)
    Xk_tr, Xk_te = build_features(tr, KEPT), build_features(te, KEPT)
    assert (Xk_te.index == yte.index).all()

    preds = {
        "mean": MeanBaseline().fit(Xr_tr, ytr).predict(Xr_te),
        "past spend": PastSpendBaseline().fit(Xr_tr, ytr).predict(Xr_te),
        "A: ratio Tweedie, R,F,M": RatioXGB(objective="tweedie", **BEST_7E_TW).fit(Xr_tr, ytr).predict(Xr_te),
        "C: ratio Tweedie, +pair (HEADLINE)": RatioXGB(objective="tweedie", **BEST_8).fit(Xk_tr, ytr).predict(Xk_te),
    }
    print(f"test customers {len(yt)}, zeros {np.mean(yt == 0):.0%}, oracle capture {oracle_capture(yt):.3f}")
    rows = {}
    for k, p in preds.items():
        e = evaluate(yt, p)
        row = {}
        for m in ("R2", "MAE", "Capture@20%"):
            lo, hi = bootstrap_ci(yt, p, m)
            row[m] = f"{e[m]:.3f} [{lo:.3f}, {hi:.3f}]"
        y2, p2 = drop_top(yt, p, 10)
        e2 = evaluate(y2, p2)
        row["R2 w/o top10"], row["MAE w/o top10"], row["Cap w/o top10"] = (round(e2["R2"], 3), round(e2["MAE"], 1), round(e2["Capture@20%"], 3))
        row["pred total / actual"] = round(p.sum() / yt.sum(), 3)
        rows[k] = row
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
    print(pd.DataFrame(rows).T.to_string())

    rng = np.random.default_rng(0)
    c, base_p, a = preds["C: ratio Tweedie, +pair (HEADLINE)"], preds["past spend"], preds["A: ratio Tweedie, R,F,M"]
    for name, other in (("C minus past spend", base_p), ("C minus A", a)):
        d = {"R2": [], "MAE gain": [], "Capture": []}
        for _ in range(1000):
            i = rng.integers(0, len(yt), len(yt))
            d["R2"].append(r2(yt[i], c[i]) - r2(yt[i], other[i]))
            d["MAE gain"].append(mae(yt[i], other[i]) - mae(yt[i], c[i]))
            d["Capture"].append(revenue_capture(yt[i], c[i]) - revenue_capture(yt[i], other[i]))
        print(f"\n{name} (bootstrap over test customers):")
        for k, v in d.items():
            print(f"  {k}: {np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]")
    top = np.argsort(-yt)[:10]
    print(f"\nlargest actual {yt.max():,.0f}; C largest prediction {c.max():,.0f}; "
          f"share of C squared error from top 10 customers {((yt - c) ** 2)[top].sum() / ((yt - c) ** 2).sum():.0%}")


if __name__ == "__main__":
    main()
