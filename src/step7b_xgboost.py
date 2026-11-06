"""Step 7b: candidate 1 of 4: XGBoost on raw GBP (squared-error objective).

Gradient-boosted trees: many small decision trees added one after another,
each correcting the previous ones' mistakes. It works directly in GBP (no log,
nothing to convert back) and can learn non-linear shapes.

Tuned with Optuna (250 trials, TPE), five hyperparameters:
  n_estimators      number of trees
  max_depth         how deep each tree can go (more depth = more complex)
  learning_rate     how big a correction each new tree makes
  subsample         share of customers each tree sees
  colsample_bytree  share of features each tree sees
Winner of the trials = highest R2 in GBP from 5-fold CV on train only.
Test is NOT touched.

Run:  python src/step7b_xgboost.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from metrics import r2
from step3_split import TRAIN_CUTOFF, load_ledger, make_snapshot
from step4_rfm import build_rfm
from step6_baselines import BASELINES, cv_predictions, score_table
from step7a_ridge_log import LogRidge
from tuning import tune

N_TRIALS = 250
ALPHA_7A_SMEAR = 4829.2  # best alpha found in step 7a (smearing variant, 250 trials), for the comparison row


class XGBModel:
    """XGBoost on raw GBP. Trees don't care about feature scale, so no log/scaling."""

    def __init__(self, **params):
        self.params = params

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model_ = XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                   random_state=42, n_jobs=4, verbosity=0, **self.params)
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model_.predict(X), 0.0, None)


def suggest_xgb(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


def oof_with_seed(make_model, X, y, seed: int) -> np.ndarray:
    """Out-of-fold predictions with a different fold assignment (stability check)."""
    oof = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        oof[va] = make_model().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va])
    return oof


def main() -> None:
    train = make_snapshot(load_ledger(), TRAIN_CUTOFF)
    X, y = build_rfm(train), train.target
    ya = y.to_numpy()

    study = tune(lambda **p: XGBModel(**p), suggest_xgb, X, y, n_trials=N_TRIALS)
    best = study.best_params
    print("best params:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()})
    print(f"best CV R2 (the tuning score, seed-42 folds): {study.best_value:.3f}")

    preds = {n: cv_predictions(c, X, y) for n, c in BASELINES.items()}
    preds["LogRidge + smearing (7a)"] = cv_predictions(lambda: LogRidge(ALPHA_7A_SMEAR, True), X, y)
    preds["XGBoost raw GBP (7b)"] = cv_predictions(lambda: XGBModel(**best), X, y)

    pd.set_option("display.width", 200)
    t = score_table(ya, preds)
    t["total pred / total actual"] = [preds[k].sum() / ya.sum() for k in t.index]
    print("\n=== TRAIN, 5-fold CV, GBP scale ===")
    print(t.round(3).to_string())
    print("\nwithout the 10 biggest customers:")
    print(score_table(ya, preds, without_top=10).round(3).to_string())

    # How optimistic is the tuning score? Same params, folds the tuner never saw.
    rs = [r2(ya, oof_with_seed(lambda: XGBModel(**best), X, y, s)) for s in range(10)]
    print(f"\nR2 with the tuned params on 10 OTHER fold assignments: mean {np.mean(rs):.3f}, "
          f"sd {np.std(rs):.3f}, min {min(rs):.3f}, max {max(rs):.3f}")

    imp = pd.Series(XGBModel(**best).fit(X, y).model_.feature_importances_, index=X.columns)
    print("feature importance (share of splits' gain):", imp.round(3).to_dict())


if __name__ == "__main__":
    main()
