"""Step 10a: P(customer spends > 0 in the next 6 months), the "likely to return" axis.

Target: 1 if the floored 6-month target > 0, else 0 (train: 48% yes).
Models, all scored by AUC (5-fold CV on train, repeated over fold seeds 0..9):
  recency only      : score = -Recency (shorter gap = more likely)
  logistic, R,F,M   : on log-scaled R, F, M
  XGB, R,F,M        : small fixed trees (depth 3, 200 trees, lr 0.05)
  XGB, + IPI/Overdue: the kept columns from step 8
Also: calibration (does 70% mean ~70%?) with Brier score and a 10-bin table.
Test set is NOT touched.

Run:  python src/step10_repurchase_segmentation.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier

from step3_split import TRAIN_CUTOFF, load_ledger, make_snapshot
from step8_features import build_features
from step8_final import KEPT

XGB_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", random_state=42, n_jobs=4, verbosity=0)


class RecencyOnly:
    def fit(self, X, y): return self
    def predict_proba(self, X): return -X["Recency"].to_numpy()  # a score, not a probability


class LogisticRFM:
    def fit(self, X, y):
        self.m_ = LogisticRegression(max_iter=1000).fit(self._t(X), y); return self
    @staticmethod
    def _t(X):
        m = X["Monetary"].to_numpy()
        return np.c_[np.log1p(X["Recency"]), np.log1p(X["Frequency"]), np.sign(m) * np.log1p(np.abs(m))]
    def predict_proba(self, X): return self.m_.predict_proba(self._t(X))[:, 1]


class XGBClf:
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y):
        self.m_ = XGBClassifier(**XGB_PARAMS).fit(X[self.cols], y); return self
    def predict_proba(self, X): return self.m_.predict_proba(X[self.cols])[:, 1]


def oof(make, X, y, seed, n_splits=5):
    out = np.zeros(len(y))
    for tr, va in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        out[va] = make().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])
    return out


def main() -> None:
    train = make_snapshot(load_ledger(), TRAIN_CUTOFF)
    X = build_features(train, KEPT)
    y = (train.target > 0).astype(int)
    rfm, kept = ["Recency", "Frequency", "Monetary"], ["Recency", "Frequency", "Monetary", *KEPT]
    models = {"recency only": RecencyOnly, "logistic R,F,M": LogisticRFM,
              "XGB R,F,M": lambda: XGBClf(rfm), "XGB R,F,M + IPI + Overdue": lambda: XGBClf(kept)}
    res, preds = {}, {}
    for k, mk in models.items():
        aucs = []
        for s in range(10):
            p = oof(mk, X, y, s); aucs.append(roc_auc_score(y, p))
            if s == 0: preds[k] = p
        res[k] = {"AUC mean": np.mean(aucs), "AUC sd": np.std(aucs)}
    print(f"customers {len(y)}, share who buy again {y.mean():.1%}")
    print(pd.DataFrame(res).T.round(4).to_string())

    rng = np.random.default_rng(0); a, b = preds["XGB R,F,M + IPI + Overdue"], preds["XGB R,F,M"]
    d = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y)); d.append(roc_auc_score(y.iloc[i], a[i]) - roc_auc_score(y.iloc[i], b[i]))
    print(f"\nAUC gain of the pair over XGB R,F,M (bootstrap): {np.mean(d):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")

    for k in ("logistic R,F,M", "XGB R,F,M + IPI + Overdue"):
        p = preds[k]
        print(f"\ncalibration, {k}: Brier {brier_score_loss(y, p):.4f} (always-the-base-rate: {brier_score_loss(y, np.full(len(y), y.mean())):.4f})")
        t = pd.DataFrame({"p": p, "y": y.to_numpy()}); t["bin"] = pd.qcut(t["p"], 10, duplicates="drop")
        print(t.groupby("bin", observed=True).agg(predicted=("p", "mean"), actual=("y", "mean"), n=("y", "size")).round(3).to_string())


if __name__ == "__main__":
    main()
