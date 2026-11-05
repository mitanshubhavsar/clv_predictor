"""Step 6: baselines. The bar every model must beat.

  1. mean       : everyone gets the train-set mean target.
  2. past_spend : predicted future spend = k * Monetary (floored at 0), with k
                  fitted on train as sum(target) / sum(Monetary). k only sets
                  the GBP scale; the RANKING is just "sort by past spend".

Both are fitted on the train snapshot only. Two ways of scoring them:
  * train 5-fold CV : fit on 4 folds, predict the held-out fold, pool the
                      out-of-fold predictions. This is the protocol step 7 uses
                      to choose between models.
  * test            : fit on all of train, score on the later test snapshot.
                      The baselines have no tuned choices, so nothing is picked
                      using this number; it is the reference for the final report.

Run:  python src/step6_baselines.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from metrics import drop_top, evaluate, oracle_capture
from step3_split import TEST_CUTOFF, TRAIN_CUTOFF, load_ledger, make_snapshot
from step4_rfm import build_rfm

SEED = 42


class MeanBaseline:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.mean_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_)


class PastSpendBaseline:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        m = X["Monetary"].clip(lower=0)
        self.k_ = float(y.sum() / m.sum())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.k_ * X["Monetary"].clip(lower=0).to_numpy()


BASELINES = {"mean": MeanBaseline, "past_spend": PastSpendBaseline}


def cv_predictions(make_model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predictions on the train customers."""
    oof = np.zeros(len(y))
    for tr, va in KFold(n_splits, shuffle=True, random_state=SEED).split(X):
        model = make_model().fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict(X.iloc[va])
    return oof


def score_table(y, preds: dict, without_top: int = 0) -> pd.DataFrame:
    """One row per model, three metrics. With without_top=10, the 10 customers
    with the largest actual spend are removed first (robustness check)."""
    rows = {}
    for name, p in preds.items():
        yy, pp = (drop_top(y, p, without_top) if without_top else (y, p))
        rows[name] = evaluate(yy, pp)
    return pd.DataFrame(rows).T


def main() -> None:
    ledger = load_ledger()
    train = make_snapshot(ledger, TRAIN_CUTOFF)
    test = make_snapshot(ledger, TEST_CUTOFF)
    Xtr, ytr = build_rfm(train), train.target
    Xte, yte = build_rfm(test), test.target

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("=== TRAIN, 5-fold CV (out-of-fold predictions) ===")
    cv = {name: cv_predictions(cls, Xtr, ytr) for name, cls in BASELINES.items()}
    print(score_table(ytr.to_numpy(), cv).round(3).to_string())
    print(f"ceiling (perfect ranking) Capture@20%: {oracle_capture(ytr.to_numpy()):.3f}")
    print("without the 10 biggest customers:")
    print(score_table(ytr.to_numpy(), cv, without_top=10).round(3).to_string())

    print("\n=== TEST (fit on all train, score the later snapshot) ===")
    fitted = {name: cls().fit(Xtr, ytr) for name, cls in BASELINES.items()}
    print("fitted: mean =", round(fitted["mean"].mean_, 1), "| past_spend k =", round(fitted["past_spend"].k_, 3))
    te = {name: m.predict(Xte) for name, m in fitted.items()}
    print(score_table(yte.to_numpy(), te).round(3).to_string())
    print(f"ceiling (perfect ranking) Capture@20%: {oracle_capture(yte.to_numpy()):.3f}")
    print("without the 10 biggest customers:")
    print(score_table(yte.to_numpy(), te, without_top=10).round(3).to_string())


if __name__ == "__main__":
    main()
