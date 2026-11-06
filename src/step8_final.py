"""Step 8 final: re-tune the ratio Tweedie model once on the kept columns
(R, F, M + IPI + OverdueRatio), 250 trials, then compare on fresh fold splits
with the 7e model (R, F, M only). Test set is NOT touched.

Run:  python src/step8_final.py [n_trials]
"""
import sys

import numpy as np
import pandas as pd

from metrics import evaluate, r2, revenue_capture
from step3_split import TRAIN_CUTOFF, load_ledger, make_snapshot
from step7b_xgboost import oof_with_seed
from step7e_ratio import BEST_7E_TW, RatioXGB, suggest_ratio
from step8_features import build_features
from tuning import tune

KEPT = ("IPI", "OverdueRatio")
# Final headline model C (re-tuned on R,F,M + IPI + OverdueRatio; 250 trials)
BEST_8 = {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.0177, "subsample": 0.5816,
          "colsample_bytree": 0.7133, "weight_power": 0.8459, "tweedie_variance_power": 1.8777}


def main(n_trials: int = 250) -> None:
    train = make_snapshot(load_ledger(), TRAIN_CUTOFF)
    y, ya = train.target, train.target.to_numpy()
    X_rfm, X_kept = build_features(train), build_features(train, KEPT)

    study = tune(lambda **p: RatioXGB(objective="tweedie", **p),
                 lambda trial: suggest_ratio(trial, True), X_kept, y, n_trials=n_trials)
    best = study.best_params
    print(f"tuning score (optimistic): {study.best_value:.3f}")
    print("best:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()})

    models = {"A: 7e settings, R,F,M": (lambda: RatioXGB(objective="tweedie", **BEST_7E_TW), X_rfm),
              "B: 7e settings, kept columns": (lambda: RatioXGB(objective="tweedie", **BEST_7E_TW), X_kept),
              "C: re-tuned, kept columns": (lambda: RatioXGB(objective="tweedie", **best), X_kept)}
    # fresh fold seeds (10..19): not used in tuning (42) or in the earlier screens (0..9)
    seeds = range(10, 20)
    preds = {k: [oof_with_seed(m, X, y, s) for s in seeds] for k, (m, X) in models.items()}
    rows = {}
    for k, ps in preds.items():
        ev = [evaluate(ya, p) for p in ps]
        rows[k] = {m: np.mean([e[m] for e in ev]) for m in ("R2", "MAE", "Capture@20%")}
        rows[k]["R2 sd"] = np.std([e["R2"] for e in ev])
    print("\nFresh fold seeds 10..19 (mean over seeds):")
    print(pd.DataFrame(rows).T.round(4).to_string())

    rng = np.random.default_rng(0)
    pa, pc = preds["A: 7e settings, R,F,M"][0], preds["C: re-tuned, kept columns"][0]
    out = {"R2": [], "MAE gain": [], "Capture": []}
    for _ in range(1000):
        i = rng.integers(0, len(ya), len(ya))
        out["R2"].append(r2(ya[i], pc[i]) - r2(ya[i], pa[i]))
        out["MAE gain"].append(np.abs(ya[i] - pa[i]).mean() - np.abs(ya[i] - pc[i]).mean())
        out["Capture"].append(revenue_capture(ya[i], pc[i]) - revenue_capture(ya[i], pa[i]))
    print("\nC minus A, bootstrap over customers (mean, 2.5%, 97.5%):")
    for k, v in out.items():
        print(f"  {k}: {np.mean(v):+.4f} [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 250)
