"""Step 8: extra features, added one at a time to the ratio model (step 7e).

build_features(snapshot, extra=("AOV", ...)) returns Recency, Frequency,
Monetary plus the requested extra columns, one row per customer, built ONLY
from the snapshot's feature rows (everything before the cutoff). The same
function serves the train and the test snapshot.

screen(): how much does adding feature(s) change the ratio model, with the 7e
settings FIXED (no re-tuning) and the SAME fold splits for both models, so the
differences are paired. Reports, per fold split: R2, MAE, Capture@20% for the
baseline (R, F, M) and for baseline + feature, and the differences.

Run:  python src/step8_features.py
"""
import numpy as np
import pandas as pd

from metrics import evaluate
from step3_split import Snapshot, TRAIN_CUTOFF, load_ledger, make_snapshot
from step4_rfm import build_rfm
from step7b_xgboost import oof_with_seed
from step7e_ratio import BEST_7E_TW, RatioXGB

N_SPLITS = 10


def aov(snap: Snapshot) -> pd.Series:
    """Average order value = gross purchase spend / number of purchase invoices.
    (Gross, not net of returns: it describes the size of an ORDER.)"""
    rows = snap.feature_rows
    p = rows[~rows["IsReturn"]]
    spend = p.groupby("Customer ID")["TotalPrice"].sum()
    orders = p.groupby("Customer ID")["Invoice"].nunique()
    return (spend / orders).rename("AOV")


FEATURES = {"AOV": aov}


def build_features(snap: Snapshot, extra=()) -> pd.DataFrame:
    X = build_rfm(snap)
    for name in extra:
        X[name] = FEATURES[name](snap).reindex(X.index)
    assert not X.isna().any().any(), "missing values in features"
    return X


def make_ratio_tweedie():
    return RatioXGB(objective="tweedie", **BEST_7E_TW)


def screen(X_base: pd.DataFrame, X_new: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """Paired comparison on the same fold splits (seeds 0..n_splits-1)."""
    ya = y.to_numpy()
    rows = []
    for seed in range(n_splits):
        a = evaluate(ya, oof_with_seed(make_ratio_tweedie, X_base, y, seed))
        b = evaluate(ya, oof_with_seed(make_ratio_tweedie, X_new, y, seed))
        rows.append({"seed": seed,
                     "R2 base": a["R2"], "R2 new": b["R2"], "R2 diff": b["R2"] - a["R2"],
                     "MAE base": a["MAE"], "MAE new": b["MAE"], "MAE gain": a["MAE"] - b["MAE"],
                     "Cap base": a["Capture@20%"], "Cap new": b["Capture@20%"], "Cap diff": b["Capture@20%"] - a["Capture@20%"]})
    return pd.DataFrame(rows).set_index("seed")


def main() -> None:
    train = make_snapshot(load_ledger(), TRAIN_CUTOFF)
    y = train.target
    X_base = build_features(train)
    X_new = build_features(train, ("AOV",))
    print("AOV summary:", X_new["AOV"].describe().round(1).to_dict())
    t = screen(X_base, X_new, y)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string())
    for col in ("R2 diff", "MAE gain", "Cap diff"):
        d = t[col]
        print(f"{col}: mean {d.mean():+.4f}, sd {d.std():.4f}, improved in {(d > 0).sum()}/{len(d)} splits, min {d.min():+.4f}, max {d.max():+.4f}")


if __name__ == "__main__":
    main()
