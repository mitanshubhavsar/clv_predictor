"""Step 7c: candidate 3 of 4: XGBoost with the Tweedie objective.

Same boosted trees as 7b, but the loss function assumes the target looks like
"many exact zeros plus a long right tail" (a Tweedie distribution), which is
exactly what our 6-month spend looks like. Differences from 7b:
  * predictions are always >= 0 (the trees work on a log scale internally and
    the model predicts exp(score)), so it never predicts negative spend;
  * errors are weighted relative to the size of the customer, instead of every
    GBP counting the same (so the giants dominate the loss less).
One extra setting: tweedie_variance_power in (1, 2).
  close to 1 -> behaves like a count of events (Poisson-like, more zero-friendly)
  close to 2 -> behaves like a heavy-tailed positive amount (Gamma-like)

Tuned with Optuna (250 trials, TPE): the five settings from 7b plus
tweedie_variance_power. Winner = highest R2 in GBP from 5-fold CV on train.
Test is NOT touched.

Run:  python src/step7c_tweedie.py
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from metrics import r2
from step3_split import TRAIN_CUTOFF, load_ledger, make_snapshot
from step4_rfm import build_rfm
from step6_baselines import BASELINES, cv_predictions, score_table
from step7a_ridge_log import LogRidge
from step7b_xgboost import XGBModel, oof_with_seed
from tuning import tune

N_TRIALS = 250
# Best settings from the earlier steps (250 trials each), for the comparison rows.
ALPHA_7A_SMEAR = 4829.2
BEST_7B = {"n_estimators": 480, "max_depth": 5, "learning_rate": 0.0111,
           "subsample": 0.3009, "colsample_bytree": 0.5686}


class XGBTweedie:
    def __init__(self, **params):
        self.params = params

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model_ = XGBRegressor(objective="reg:tweedie", tree_method="hist",
                                   random_state=42, n_jobs=4, verbosity=0, **self.params)
        self.model_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict(X)  # already >= 0 (exp of the score)


def suggest_tweedie(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.01, 1.99),
    }


def main() -> None:
    train = make_snapshot(load_ledger(), TRAIN_CUTOFF)
    X, y = build_rfm(train), train.target
    ya = y.to_numpy()

    study = tune(lambda **p: XGBTweedie(**p), suggest_tweedie, X, y, n_trials=N_TRIALS)
    best = study.best_params
    print("best params:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()})
    print(f"best CV R2 (tuning score, seed-42 folds): {study.best_value:.3f}")

    preds = {n: cv_predictions(c, X, y) for n, c in BASELINES.items()}
    preds["LogRidge + smearing (7a)"] = cv_predictions(lambda: LogRidge(ALPHA_7A_SMEAR, True), X, y)
    preds["XGBoost raw GBP (7b)"] = cv_predictions(lambda: XGBModel(**BEST_7B), X, y)
    preds["XGBoost Tweedie (7c)"] = cv_predictions(lambda: XGBTweedie(**best), X, y)

    pd.set_option("display.width", 200)
    t = score_table(ya, preds)
    t["total pred / total actual"] = [preds[k].sum() / ya.sum() for k in t.index]
    print("\n=== TRAIN, 5-fold CV, GBP scale ===")
    print(t.round(3).to_string())
    print("\nwithout the 10 biggest customers:")
    print(score_table(ya, preds, without_top=10).round(3).to_string())

    rs = [r2(ya, oof_with_seed(lambda: XGBTweedie(**best), X, y, s)) for s in range(10)]
    print(f"\nR2 with the tuned params on 10 OTHER fold assignments: mean {np.mean(rs):.3f}, "
          f"sd {np.std(rs):.3f}, min {min(rs):.3f}, max {max(rs):.3f}")

    tw = preds["XGBoost Tweedie (7c)"]
    print(f"largest Tweedie prediction: {tw.max():,.0f} | largest actual: {ya.max():,.0f}")
    top = np.argsort(-ya)[:10]
    for n in ("past_spend", "XGBoost raw GBP (7b)", "XGBoost Tweedie (7c)"):
        e = (ya - preds[n]) ** 2
        print(f"{n:22s} share of squared error from the 10 biggest actual customers: {e[top].sum() / e.sum():.0%}")
    imp = pd.Series(XGBTweedie(**best).fit(X, y).model_.feature_importances_, index=X.columns)
    print("feature importance:", imp.round(3).to_dict())


if __name__ == "__main__":
    main()
