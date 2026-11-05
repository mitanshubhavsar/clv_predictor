"""Step 5: look closely at the target before modelling. Analysis only: the
target is NOT changed here; decisions are taken after reading these numbers.

Questions:
  1. How is the target made up? (zeros / negatives / positives)
  2. Who are the negative-target customers, and what would flooring at 0 change?
  3. How skewed is it, and how concentrated is the money?
  4. Is "buys again or not" predictable from RFM? (does a zero-vs-nonzero split make sense)

Note: decisions taken after this analysis are already applied in step 3:
negatives floored at 0 (snap.target; snap.target_net keeps the original),
no clipping of large customers. Sections 1-2 therefore look at target_net.

Run:  python src/step5_target.py
"""
import numpy as np
import pandas as pd

from step3_split import (Snapshot, TEST_CUTOFF, TRAIN_CUTOFF, load_ledger,
                         make_snapshot)
from step4_rfm import build_rfm


def composition(t: pd.Series) -> pd.Series:
    return pd.Series({"customers": len(t), "zero": (t == 0).sum(),
                      "negative": (t < 0).sum(), "positive": (t > 0).sum()})


def negative_customers(snap: Snapshot, ledger: pd.DataFrame) -> pd.DataFrame:
    """For negative-target customers: money bought vs returned in the target window."""
    neg = snap.target_net[snap.target_net < 0].index
    w = ledger[(ledger["InvoiceDate"] >= snap.cutoff) & (ledger["InvoiceDate"] < snap.target_end)
               & ledger["Customer ID"].isin(neg)]
    out = w.groupby(["Customer ID", "IsReturn"])["TotalPrice"].sum().unstack(fill_value=0.0)
    out.columns = ["bought_in_window", "returned_in_window"]
    out["net"] = out.sum(axis=1)
    return out


def concentration(t: pd.Series) -> dict:
    pos = t[t > 0].sort_values(ascending=False)
    total = pos.sum()
    return {f"top {p}% of positive-target customers": pos.head(max(1, int(len(pos) * p / 100))).sum() / total
            for p in (1, 5, 10, 20)}


def repurchase_rates(rfm: pd.DataFrame, t: pd.Series) -> pd.DataFrame:
    """P(target > 0) by Frequency bucket and by Recency quartile."""
    buys = (t > 0)
    f = pd.cut(rfm["Frequency"], [0, 1, 2, 5, 10, np.inf], labels=["1", "2", "3-5", "6-10", "11+"])
    r = pd.qcut(rfm["Recency"], 4, labels=["R1 most recent", "R2", "R3", "R4 least recent"])
    return pd.concat({"Frequency": buys.groupby(f, observed=True).agg(["mean", "size"]),
                      "Recency": buys.groupby(r, observed=True).agg(["mean", "size"])})


def main() -> None:
    ledger = load_ledger()
    for name, cutoff in [("TRAIN", TRAIN_CUTOFF), ("TEST", TEST_CUTOFF)]:
        snap = make_snapshot(ledger, cutoff)
        t, rfm = snap.target, build_rfm(snap)
        print(f"\n================ {name} ================")
        print("1. composition (before flooring):", composition(snap.target_net).to_dict())
        pos = t[t > 0]
        print("   positive targets, percentiles:",
              pos.quantile([.1, .25, .5, .75, .9, .99]).round(0).to_dict())

        print("\n2. negative targets")
        neg = negative_customers(snap, ledger)
        print(neg.describe().round(0).loc[["count", "mean", "min", "max"]].to_string())
        tn = snap.target_net
        print(f"   sum of negatives: {tn[tn < 0].sum():,.0f} | mean target unfloored {tn.mean():,.1f}"
              f" vs floored at 0 {t.mean():,.1f}")

        print("\n3. skew and concentration")
        print(f"   skewness (floored) {t.skew():.1f} | log1p(floored) {np.log1p(t).skew():.2f}")
        for k, v in concentration(t).items():
            print(f"   {k}: {v:.0%} of money")
        top10 = t.nlargest(10).index
        print(f"   mean target {t.mean():,.0f} -> without top 10 customers {t.drop(top10).mean():,.0f}")

        print("\n4. P(buys again) by RFM bucket")
        print(repurchase_rates(rfm, t).round(2).to_string())


if __name__ == "__main__":
    main()
