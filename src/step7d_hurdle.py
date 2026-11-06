"""Step 7d: candidate 4 of 4: the Hurdle model (two stages).

  Stage 1 (classifier): P(customer buys at all in the next 6 months).
  Stage 2 (regressor) : expected spend GIVEN they buy, trained ONLY on the
                        customers who did buy (so no zeros in its target).
  Prediction          : P(buy) x E[spend | buy].

Two variants of stage 2 (as in 7a, plain vs smearing):
  raw        : XGBoost regressor on raw GBP (squared error)
  log+smear  : XGBoost regressor on log1p(GBP), converted back with Duan
               smearing. Among buyers there are no zeros, so the log residuals
               are far smaller than in 7a and the smearing factor is stable.

Both stages are XGBoost, tuned JOINTLY with Optuna (10 settings: the usual five
for each stage, prefixed c_ and r_), 250 trials, winner = highest R2 in GBP
from 5-fold CV on train. Test is NOT touched.

Run:  python src/step7d_hurdle.py [n_trials]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor

from metrics import r2
from step3_split import TRAIN_CUTOFF, load_ledger, make_snapshot
from step4_rfm import build_rfm
from step6_baselines import BASELINES, cv_predictions, score_table
from step7a_ridge_log import LogRidge
from step7b_xgboost import XGBModel, oof_with_seed
from step7c_tweedie import XGBTweedie
from tuning import tune

N_TRIALS = 250
# Best settings from the earlier steps (250 trials each), for the comparison rows.
ALPHA_7A_SMEAR = 4829.2
BEST_7B = {"n_estimators": 480, "max_depth": 5, "learning_rate": 0.0111,
           "subsample": 0.3009, "colsample_bytree": 0.5686}
BEST_7C = {"n_estimators": 395, "max_depth": 2, "learning_rate": 0.0141,
           "subsample": 0.9957, "colsample_bytree": 0.7604, "tweedie_variance_power": 1.5677}

# Tuned Hurdle settings from this step (250 trials each), kept for later comparison steps.
BEST_7D_RAW = {"c_n_estimators": 419, "c_max_depth": 2, "c_learning_rate": 0.1918, "c_subsample": 0.5442, "c_colsample_bytree": 0.9493,
               "r_n_estimators": 152, "r_max_depth": 2, "r_learning_rate": 0.0302, "r_subsample": 0.3005, "r_colsample_bytree": 0.6253}
BEST_7D_LOG = {"c_n_estimators": 409, "c_max_depth": 3, "c_learning_rate": 0.0191, "c_subsample": 0.8474, "c_colsample_bytree": 0.7053,
               "r_n_estimators": 487, "r_max_depth": 2, "r_learning_rate": 0.0283, "r_subsample": 0.4383, "r_colsample_bytree": 0.736}

STAGE_KEYS = ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree")


class HurdleModel:
    def __init__(self, log_stage2: bool = False, **params):
        self.log_stage2 = log_stage2
        self.clf_params = {k[2:]: v for k, v in params.items() if k.startswith("c_")}
        self.reg_params = {k[2:]: v for k, v in params.items() if k.startswith("r_")}
        self.params = params

    def fit(self, X: pd.DataFrame, y: pd.Series):
        yb = (y > 0).astype(int)
        self.clf_ = XGBClassifier(tree_method="hist", random_state=42, n_jobs=4,
                                  verbosity=0, **self.clf_params).fit(X, yb)
        pos = (y > 0).to_numpy()
        target = np.log1p(y[pos]) if self.log_stage2 else y[pos]
        self.reg_ = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42,
                                 n_jobs=4, verbosity=0, **self.reg_params).fit(X[pos], target)
        if self.log_stage2:  # Duan smearing factor, from the buyers only
            resid = target.to_numpy() - self.reg_.predict(X[pos])
            self.smear_ = float(np.mean(np.exp(resid)))
        return self

    def predict_parts(self, X: pd.DataFrame):
        p = self.clf_.predict_proba(X)[:, 1]
        raw = self.reg_.predict(X)
        amount = np.exp(raw) * self.smear_ - 1.0 if self.log_stage2 else raw
        return p, np.clip(amount, 0.0, None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p, amount = self.predict_parts(X)
        return p * amount


def suggest_hurdle(trial):
    out = {}
    for prefix in ("c_", "r_"):
        out[prefix + "n_estimators"] = trial.suggest_int(prefix + "n_estimators", 50, 500)
        out[prefix + "max_depth"] = trial.suggest_int(prefix + "max_depth", 2, 6)
        out[prefix + "learning_rate"] = trial.suggest_float(prefix + "learning_rate", 0.01, 0.3, log=True)
        out[prefix + "subsample"] = trial.suggest_float(prefix + "subsample", 0.3, 1.0)
        out[prefix + "colsample_bytree"] = trial.suggest_float(prefix + "colsample_bytree", 0.5, 1.0)
    return out


def oof_parts(make_model, X, y, seed: int = 42):
    """Out-of-fold P(buy) and E[spend | buy] for the whole train set."""
    p, a = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(X):
        m = make_model().fit(X.iloc[tr], y.iloc[tr])
        p[va], a[va] = m.predict_parts(X.iloc[va])
    return p, a


def main(n_trials: int = N_TRIALS) -> None:
    train = make_snapshot(load_ledger(), TRAIN_CUTOFF)
    X, y = build_rfm(train), train.target
    ya = y.to_numpy()

    preds = {n: cv_predictions(c, X, y) for n, c in BASELINES.items()}
    preds["LogRidge + smearing (7a)"] = cv_predictions(lambda: LogRidge(ALPHA_7A_SMEAR, True), X, y)
    preds["XGBoost raw GBP (7b)"] = cv_predictions(lambda: XGBModel(**BEST_7B), X, y)
    preds["XGBoost Tweedie (7c)"] = cv_predictions(lambda: XGBTweedie(**BEST_7C), X, y)

    best, honest = {}, {}
    for log2, label in ((False, "Hurdle, stage 2 raw GBP (7d)"), (True, "Hurdle, stage 2 log+smear (7d)")):
        study = tune(lambda log2=log2, **p: HurdleModel(log_stage2=log2, **p), suggest_hurdle, X, y, n_trials=n_trials)
        best[label] = study.best_params
        print(f"\n{label}: tuning score {study.best_value:.3f}")
        print("  stage 1:", {k[2:]: round(v, 4) if isinstance(v, float) else v for k, v in study.best_params.items() if k.startswith("c_")})
        print("  stage 2:", {k[2:]: round(v, 4) if isinstance(v, float) else v for k, v in study.best_params.items() if k.startswith("r_")})
        preds[label] = cv_predictions(lambda log2=log2, b=study.best_params: HurdleModel(log_stage2=log2, **b), X, y)
        rs = [r2(ya, oof_with_seed(lambda log2=log2, b=study.best_params: HurdleModel(log_stage2=log2, **b), X, y, s)) for s in range(10)]
        honest[label] = rs

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
    print("\nlargest prediction vs largest actual", f"{ya.max():,.0f}:")
    for k in best:
        e = (ya - preds[k]) ** 2
        print(f"  {k}: largest {preds[k].max():,.0f} | share of squared error from 10 biggest customers {e[top].sum() / e.sum():.0%}")

    # Where does the loss come from: stage 1 (who buys) or stage 2 (how much)?
    print("\nStage diagnostics (out-of-fold):")
    buys = ya > 0
    for k, log2 in (("Hurdle, stage 2 raw GBP (7d)", False), ("Hurdle, stage 2 log+smear (7d)", True)):
        p, a = oof_parts(lambda log2=log2, b=best[k]: HurdleModel(log_stage2=log2, **b), X, y)
        print(f"  {k}")
        print(f"    stage 1 AUC (who buys): {roc_auc_score(buys, p):.3f} | mean P(buy) {p.mean():.3f} vs actual share {buys.mean():.3f}")
        print(f"    stage 2 among real buyers: R2 {r2(ya[buys], a[buys]):.3f}, MAE {np.mean(np.abs(ya[buys] - a[buys])):,.0f}")
        print(f"    R2 if stage 1 were perfect (true 0/1 x stage 2): {r2(ya, buys * a):.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_TRIALS)
