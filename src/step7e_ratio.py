"""Step 7e: the RATIO idea: predict a multiplier, not pounds.

  scale      = max(Monetary, FLOOR)         (the customer's own past spend)
  multiplier = future spend / scale         (what the trees learn)
  prediction = multiplier_hat x scale

Why: trees cannot output numbers larger than the averages they were trained on
(see model_tried_summary.md, "why trees struggle with the giants"). Here the
SCALE comes from the customer, so a customer with 300,000 past spend gets a big
prediction even if the trees only ever learned "multiplier is about 0.3".
A single constant multiplier IS the past-spend baseline (0.37 x Monetary), so
the model can only improve on it by correcting it with Recency / Frequency / ...

FLOOR: a tiny past spend with a real future spend would give a huge multiplier,
and Monetary can be <= 0. So the scale never goes below FLOOR (50 GBP).

Weights: customers are weighted by scale ** weight_power (normalised to mean 1).
  power 2 -> the fit minimises squared error in GBP exactly (big customers rule);
  power 0 -> every customer's multiplier counts the same;
  Optuna chooses weight_power in [0, 2].

Two variants (same boosted trees, different loss):
  squared : reg:squarederror on the multiplier
  tweedie : reg:tweedie on the multiplier (zeros allowed, never negative)

Tuned with Optuna (250 trials each): the usual five settings + weight_power
(+ tweedie_variance_power). Winner = highest R2 in GBP from 5-fold CV on train.
Test is NOT touched.

Run:  python src/step7e_ratio.py [n_trials]
"""
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from metrics import r2
from step3_split import TRAIN_CUTOFF, load_ledger, make_snapshot
from step4_rfm import build_rfm
from step6_baselines import BASELINES, PastSpendBaseline, cv_predictions, score_table
from step7a_ridge_log import LogRidge
from step7b_xgboost import XGBModel, oof_with_seed
from step7c_tweedie import XGBTweedie
from step7d_hurdle import BEST_7D_LOG, HurdleModel
from tuning import tune

N_TRIALS = 250
FLOOR = 50.0
# Best settings from the earlier steps (250 trials each), for the comparison rows.
ALPHA_7A_SMEAR = 4829.2
BEST_7B = {"n_estimators": 480, "max_depth": 5, "learning_rate": 0.0111,
           "subsample": 0.3009, "colsample_bytree": 0.5686}
BEST_7C = {"n_estimators": 395, "max_depth": 2, "learning_rate": 0.0141,
           "subsample": 0.9957, "colsample_bytree": 0.7604, "tweedie_variance_power": 1.5677}


# Tuned ratio-model settings from this step (250 trials each), kept for later steps.
BEST_7E_SQ = {"n_estimators": 50, "max_depth": 5, "learning_rate": 0.0107, "subsample": 0.3408,
              "colsample_bytree": 0.9598, "weight_power": 0.5568}
BEST_7E_TW = {"n_estimators": 188, "max_depth": 2, "learning_rate": 0.0119, "subsample": 0.7711,
              "colsample_bytree": 0.6706, "weight_power": 0.8569, "tweedie_variance_power": 1.5008}


class RatioXGB:
    def __init__(self, objective: str = "squared", weight_power: float = 2.0, **params):
        self.objective, self.weight_power, self.params = objective, weight_power, params

    @staticmethod
    def scale(X: pd.DataFrame) -> np.ndarray:
        return X["Monetary"].clip(lower=FLOOR).to_numpy()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        s = self.scale(X)
        w = s ** self.weight_power
        w = w / w.mean()
        obj = "reg:tweedie" if self.objective == "tweedie" else "reg:squarederror"
        self.model_ = XGBRegressor(objective=obj, tree_method="hist", random_state=42,
                                   n_jobs=4, verbosity=0, **self.params)
        self.model_.fit(X, y.to_numpy() / s, sample_weight=w)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model_.predict(X) * self.scale(X), 0.0, None)


def suggest_ratio(trial, tweedie: bool = False):
    p = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "weight_power": trial.suggest_float("weight_power", 0.0, 2.0),
    }
    if tweedie:
        p["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.01, 1.99)
    return p


def main(n_trials: int = N_TRIALS) -> None:
    train = make_snapshot(load_ledger(), TRAIN_CUTOFF)
    X, y = build_rfm(train), train.target
    ya = y.to_numpy()
    print(f"customers with Monetary below the floor ({FLOOR:.0f}): {(X['Monetary'] < FLOOR).sum()} "
          f"(of which <= 0: {(X['Monetary'] <= 0).sum()})")

    preds = {n: cv_predictions(c, X, y) for n, c in BASELINES.items()}
    preds["LogRidge + smearing (7a)"] = cv_predictions(lambda: LogRidge(ALPHA_7A_SMEAR, True), X, y)
    preds["XGBoost raw GBP (7b)"] = cv_predictions(lambda: XGBModel(**BEST_7B), X, y)
    preds["XGBoost Tweedie (7c)"] = cv_predictions(lambda: XGBTweedie(**BEST_7C), X, y)
    preds["Hurdle log+smear (7d)"] = cv_predictions(lambda: HurdleModel(log_stage2=True, **BEST_7D_LOG), X, y)

    best, honest = {}, {}
    for obj, label in (("squared", "Ratio XGBoost, squared error (7e)"), ("tweedie", "Ratio XGBoost, Tweedie (7e)")):
        tw = obj == "tweedie"
        study = tune(lambda obj=obj, **p: RatioXGB(objective=obj, **p),
                     lambda trial, tw=tw: suggest_ratio(trial, tw), X, y, n_trials=n_trials)
        best[label] = study.best_params
        print(f"\n{label}: tuning score {study.best_value:.3f}")
        print("  best:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in study.best_params.items()})
        make = lambda obj=obj, b=study.best_params: RatioXGB(objective=obj, **b)
        preds[label] = cv_predictions(make, X, y)
        honest[label] = [r2(ya, oof_with_seed(make, X, y, s)) for s in range(10)]

    pd.set_option("display.width", 220)
    t = score_table(ya, preds)
    t["total pred / total actual"] = [preds[k].sum() / ya.sum() for k in t.index]
    print("\n=== TRAIN, 5-fold CV, GBP scale ===")
    print(t.round(3).to_string())
    print("\nwithout the 10 biggest customers:")
    print(score_table(ya, preds, without_top=10).round(3).to_string())
    print("\nR2 with the tuned params on 10 OTHER fold assignments:")
    for k, rs in honest.items():
        print(f"  {k}: mean {np.mean(rs):.3f}, sd {np.std(rs):.3f}, range {min(rs):.3f} to {max(rs):.3f}")

    top = np.argsort(-ya)[:10]
    print(f"\nlargest prediction vs largest actual {ya.max():,.0f}; share of squared error from the 10 biggest customers:")
    for k in ("past_spend", "XGBoost raw GBP (7b)", "Hurdle log+smear (7d)", *best):
        e = (ya - preds[k]) ** 2
        print(f"  {k:36s} largest {preds[k].max():>9,.0f} | {e[top].sum() / e.sum():.0%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_TRIALS)
