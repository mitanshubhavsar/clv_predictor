"""Step 7a: candidate 2 of 4 (the simplest): Ridge regression on log1p(target).

Pipeline:  log1p of the features  ->  StandardScaler  ->  Ridge  ->  predict
log1p(target)  ->  convert back to GBP  ->  score in GBP.

Converting back is the catch. exp() of the average log is a TYPICAL value, not
the AVERAGE: for a right-skewed target (Jensen's inequality) it comes out too
low. Duan's smearing fixes that by multiplying by the mean of exp(training
residuals). We tune and report both variants so the effect is visible.

Only the train snapshot is used (5-fold CV, R2 in GBP picks alpha). Test is
NOT touched until a winner is chosen.

Run:  python src/step7a_ridge_log.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from metrics import evaluate
from step3_split import TRAIN_CUTOFF, load_ledger, make_snapshot
from step4_rfm import build_rfm
from step6_baselines import BASELINES, cv_predictions, score_table
from tuning import tune

N_TRIALS = 250


class LogRidge:
    def __init__(self, alpha: float = 1.0, smear: bool = False):
        self.alpha, self.smear = alpha, smear

    @staticmethod
    def _prep(X: pd.DataFrame) -> np.ndarray:
        return np.log1p(X.clip(lower=0).to_numpy())  # R, F, M are skewed: log them

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.scaler_ = StandardScaler().fit(self._prep(X))
        z = np.log1p(y.to_numpy())
        self.ridge_ = Ridge(alpha=self.alpha).fit(self.scaler_.transform(self._prep(X)), z)
        # Duan smearing factor: mean of exp(residual) in log space
        resid = z - self.ridge_.predict(self.scaler_.transform(self._prep(X)))
        self.smear_ = float(np.mean(np.exp(resid))) if self.smear else 1.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        p = self.ridge_.predict(self.scaler_.transform(self._prep(X)))
        return np.clip(np.exp(p) * self.smear_ - 1.0, 0.0, None)  # back to GBP


def suggest_alpha(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-3, 1e6, log=True)}


def main() -> None:
    train = make_snapshot(load_ledger(), TRAIN_CUTOFF)
    X, y = build_rfm(train), train.target

    preds, notes = {}, {}
    for name, cls in BASELINES.items():
        preds[name] = cv_predictions(cls, X, y)

    for smear in (False, True):
        label = "LogRidge + smearing" if smear else "LogRidge (plain expm1)"
        study = tune(lambda alpha, s=smear: LogRidge(alpha=alpha, smear=s),
                     suggest_alpha, X, y, n_trials=N_TRIALS)
        best = study.best_params["alpha"]
        preds[label] = cv_predictions(lambda b=best, s=smear: LogRidge(alpha=b, smear=s), X, y)
        notes[label] = best
        if smear:
            print("smearing factor S on the full train fit:",
                  round(LogRidge(alpha=best, smear=True).fit(X, y).smear_, 2))

    pd.set_option("display.width", 200)
    print("=== TRAIN, 5-fold CV, GBP scale ===")
    t = score_table(y.to_numpy(), preds)
    t["total pred / total actual"] = [preds[k].sum() / y.sum() for k in t.index]
    print(t.round(3).to_string())
    print("best alpha:", {k: round(v, 3) for k, v in notes.items()})
    print("\nwithout the 10 biggest customers:")
    print(score_table(y.to_numpy(), preds, without_top=10).round(3).to_string())


if __name__ == "__main__":
    main()
