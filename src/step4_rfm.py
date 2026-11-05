"""Step 4: RFM features, one row per customer, built from a snapshot's feature rows.

  Recency   : days from the customer's LAST PURCHASE to the cutoff date
              (small = bought recently). Measured from the cutoff, not from
              today, so train and test snapshots are comparable.
  Frequency : number of distinct purchase invoices in the feature window
              (returns do not count as purchases).
  Monetary  : NET money kept in the feature window (purchases minus returns,
              from the ledger; same definition as the target).

Run:  python src/step4_rfm.py
"""
import pandas as pd

from step3_split import (Snapshot, TEST_CUTOFF, TRAIN_CUTOFF, load_ledger,
                         make_snapshot)


def build_rfm(snap: Snapshot) -> pd.DataFrame:
    rows = snap.feature_rows
    purchases = rows[~rows["IsReturn"]]

    last_purchase = purchases.groupby("Customer ID")["InvoiceDate"].max()
    # normalize(): count whole days, so a purchase on the day before the cutoff = 1
    recency = (snap.cutoff - last_purchase.dt.normalize()).dt.days.rename("Recency")

    frequency = purchases.groupby("Customer ID")["Invoice"].nunique().rename("Frequency")
    monetary = rows.groupby("Customer ID")["TotalPrice"].sum().rename("Monetary")

    rfm = pd.concat([recency, frequency, monetary], axis=1)
    # Same customers as the target, same order
    assert rfm.index.isin(snap.target.index).all() and len(rfm) == len(snap.target)
    return rfm.reindex(snap.target.index)


def main() -> None:
    ledger = load_ledger()
    for name, cutoff in [("TRAIN", TRAIN_CUTOFF), ("TEST", TEST_CUTOFF)]:
        snap = make_snapshot(ledger, cutoff)
        rfm = build_rfm(snap)
        print(f"\n=== {name} (cutoff {cutoff.date()}) ===")
        print(f"customers: {len(rfm):,} | missing values: {int(rfm.isna().sum().sum())}")
        print(rfm.describe().round(1).to_string())
        print("customers with Monetary <= 0 :", int((rfm["Monetary"] <= 0).sum()))
        # Spearman = rank correlation, robust to the heavy skew
        corr = rfm.assign(target=snap.target).corr(method="spearman")["target"].drop("target")
        print("Spearman correlation with target:", corr.round(3).to_dict())


if __name__ == "__main__":
    main()
