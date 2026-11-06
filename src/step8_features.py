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

from metrics import evaluate, r2, revenue_capture
from step3_split import Snapshot, TRAIN_CUTOFF, load_ledger, make_snapshot
from step4_rfm import build_rfm
from step7b_xgboost import oof_with_seed
from step7e_ratio import BEST_7E_TW, RatioXGB

N_SPLITS = 10


def aov(snap: Snapshot, X: pd.DataFrame) -> pd.Series:
    """Average order value = gross purchase spend / number of purchase invoices.
    (Gross, not net of returns: it describes the size of an ORDER.)"""
    rows = snap.feature_rows
    p = rows[~rows["IsReturn"]]
    spend = p.groupby("Customer ID")["TotalPrice"].sum()
    orders = p.groupby("Customer ID")["Invoice"].nunique()
    return (spend / orders).rename("AOV")


def ipi(snap: Snapshot, X: pd.DataFrame) -> pd.Series:
    """Purchase interval = mean days between consecutive purchase DAYS.
    Customers who bought on only one day have no interval: left as NaN, which
    XGBoost handles natively (no fill value, so nothing to leak from train to
    test; Frequency == 1 already flags these customers)."""
    rows = snap.feature_rows
    d = rows[~rows["IsReturn"]][["Customer ID", "InvoiceDate"]].copy()
    d["InvoiceDate"] = d["InvoiceDate"].dt.normalize()
    d = d.drop_duplicates().sort_values(["Customer ID", "InvoiceDate"])
    gap = d.groupby("Customer ID")["InvoiceDate"].diff().dt.days
    return gap.groupby(d["Customer ID"]).mean().rename("IPI")  # NaN if a single purchase day


def overdue_ratio(snap: Snapshot, X: pd.DataFrame) -> pd.Series:
    """Recency / interval: above 1 = the customer is later than their own usual rhythm."""
    interval = ipi(snap, X).reindex(X.index)
    return (X["Recency"] / interval.clip(lower=1.0)).rename("OverdueRatio")


RECENT_DAYS = 90


def recent_share(snap: Snapshot, X: pd.DataFrame) -> pd.Series:
    """Spend trend: share of the customer's gross purchase spend that falls in the
    last RECENT_DAYS days before the cutoff (0 = nothing recent, 1 = all recent)."""
    rows = snap.feature_rows
    p = rows[~rows["IsReturn"]]
    recent = p[p["InvoiceDate"] >= snap.cutoff - pd.Timedelta(days=RECENT_DAYS)]
    total = p.groupby("Customer ID")["TotalPrice"].sum()
    part = recent.groupby("Customer ID")["TotalPrice"].sum().reindex(total.index).fillna(0.0)
    return (part / total.where(total > 0)).rename("RecentShare")


def return_rate(snap: Snapshot, X: pd.DataFrame) -> pd.Series:
    """Return invoices / purchase invoices, inside the feature window only."""
    rows = snap.feature_rows
    n_buy = rows[~rows["IsReturn"]].groupby("Customer ID")["Invoice"].nunique()
    n_ret = rows[rows["IsReturn"]].groupby("Customer ID")["Invoice"].nunique()
    return (n_ret.reindex(n_buy.index).fillna(0) / n_buy).rename("ReturnRate")


def tenure(snap: Snapshot, X: pd.DataFrame) -> pd.Series:
    """Days from the customer's first purchase INSIDE the feature window to the cutoff
    (so capped at the window length: it cannot see older history)."""
    p = snap.feature_rows[~snap.feature_rows["IsReturn"]]
    first = p.groupby("Customer ID")["InvoiceDate"].min()
    return ((snap.cutoff - first).dt.days).rename("Tenure")


def n_products(snap: Snapshot, X: pd.DataFrame) -> pd.Series:
    """Product diversity: number of distinct stock codes bought in the window."""
    p = snap.feature_rows[~snap.feature_rows["IsReturn"]]
    return p.groupby("Customer ID")["StockCode"].nunique().rename("NProducts")


FEATURES = {"Tenure": tenure, "NProducts": n_products, "AOV": aov, "IPI": ipi, "OverdueRatio": overdue_ratio,
            "RecentShare": recent_share, "ReturnRate": return_rate}
MAY_BE_NAN = {"IPI", "OverdueRatio", "RecentShare"}


def build_features(snap: Snapshot, extra=()) -> pd.DataFrame:
    X = build_rfm(snap)
    for name in extra:
        X[name] = FEATURES[name](snap, X).reindex(X.index)
    must_be_complete = [c for c in X.columns if c not in MAY_BE_NAN]
    assert not X[must_be_complete].isna().any().any(), "missing values in features"
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


def summarize(t: pd.DataFrame) -> pd.DataFrame:
    """Mean, spread, splits improved and paired t for the three paired columns."""
    rows = {}
    for col, label in (("R2 diff", "R2"), ("MAE gain", "MAE gain (GBP)"), ("Cap diff", "Capture@20%")):
        d = t[col]
        rows[label] = {"mean": d.mean(), "sd": d.std(), "improved": int((d > 0).sum()),
                       "t": d.mean() / (d.std() / np.sqrt(len(d)))}
    return pd.DataFrame(rows).T


def customer_bootstrap(X_base, X_new, y, seed: int = 0, n_boot: int = 1000) -> dict:
    """Paired bootstrap OVER CUSTOMERS on one set of out-of-fold predictions.
    The 10 fold splits reuse the same customers, so their spread only measures
    fold-assignment noise; this measures the noise from WHICH customers we have.
    Returns {metric: (mean, low, high)} for new minus base (MAE as a gain)."""
    ya = y.to_numpy()
    pb = oof_with_seed(make_ratio_tweedie, X_base, y, seed)
    pn = oof_with_seed(make_ratio_tweedie, X_new, y, seed)
    rng = np.random.default_rng(seed)
    out = {"R2": [], "MAE gain (GBP)": [], "Capture@20%": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(ya), len(ya))
        out["R2"].append(r2(ya[i], pn[i]) - r2(ya[i], pb[i]))
        out["MAE gain (GBP)"].append(np.abs(ya[i] - pb[i]).mean() - np.abs(ya[i] - pn[i]).mean())
        out["Capture@20%"].append(revenue_capture(ya[i], pn[i]) - revenue_capture(ya[i], pb[i]))
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def rule_verdicts(t: pd.DataFrame, boot: dict | None = None) -> dict:
    """What each candidate keep rule says about one feature (True = keep).
    A, B, C use only the 10 fold splits. D, E also need the customer bootstrap."""
    s = summarize(t)
    r2_n, r2_mean, r2_t = s.loc["R2", "improved"], s.loc["R2", "mean"], s.loc["R2", "t"]
    mae_t = s.loc["MAE gain (GBP)", "t"]
    v = {"A: R2 better in >= 9/10 splits": bool(r2_n >= 9),
         "B: R2 better in >= 8/10 splits": bool(r2_n >= 8),
         "C: R2 mean > 0, t >= 2, MAE not clearly worse": bool(r2_mean > 0 and r2_t >= 2 and mae_t > -2)}
    if boot is not None:
        sig_better = [m for m in ("R2", "MAE gain (GBP)") if boot[m][1] > 0]
        sig_worse = [m for m in ("R2", "MAE gain (GBP)") if boot[m][2] < 0]
        v["D: R2 or MAE significantly better over customers, none significantly worse"] = bool(sig_better and not sig_worse)
        practical = boot["MAE gain (GBP)"][0] >= 5.0 or boot["R2"][0] >= 0.005
        v["E: D and a practical size (MAE gain >= 5 GBP or R2 >= +0.005)"] = bool(sig_better and not sig_worse and practical)
    return v


def main() -> None:
    train = make_snapshot(load_ledger(), TRAIN_CUTOFF)
    y = train.target
    X_base = build_features(train)
    pd.set_option("display.width", 200)
    for extra in (("AOV",), ("IPI",), ("OverdueRatio",), ("IPI", "OverdueRatio")):
        X_new = build_features(train, extra)
        t = screen(X_base, X_new, y)
        print(f"\n=== + {' + '.join(extra)} ===")
        print(summarize(t).round(4).to_string())
        print(rule_verdicts(t))


if __name__ == "__main__":
    main()
