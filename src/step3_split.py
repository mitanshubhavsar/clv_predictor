"""Step 3: the time split (option A: 12-month features, 6-month target).

One function, make_snapshot(ledger, cutoff), returns:
  feature_rows : ledger rows in [cutoff - 12 months, cutoff)  for the customers
                 who bought in that window
  target       : per-customer NET money kept in [cutoff, cutoff + 6 months)
                 (0 if the customer did not buy again)

Windows are half-open: the cutoff day itself belongs to the TARGET window, so
nothing on or after the cutoff can leak into the features.

Two snapshots, with non-overlapping target windows:
  train cutoff 2010-12-01 : features 2009-12-01..2010-11-30, target 2010-12-01..2011-05-31
  test  cutoff 2011-06-01 : features 2010-06-01..2011-05-31, target 2011-06-01..2011-11-30

Run:  python src/step3_split.py
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "data" / "ledger.csv"

FEATURE_MONTHS = 12
TARGET_MONTHS = 6
TRAIN_CUTOFF = pd.Timestamp("2010-12-01")
TEST_CUTOFF = pd.Timestamp("2011-06-01")


@dataclass
class Snapshot:
    cutoff: pd.Timestamp
    feature_start: pd.Timestamp
    target_end: pd.Timestamp          # exclusive
    feature_rows: pd.DataFrame        # ledger rows (purchases + returns) before cutoff
    target: pd.Series                 # index = Customer ID, value = net money kept


def load_ledger() -> pd.DataFrame:
    return pd.read_csv(LEDGER, parse_dates=["InvoiceDate"])


def make_snapshot(ledger: pd.DataFrame, cutoff: pd.Timestamp,
                  feature_months: int = FEATURE_MONTHS,
                  target_months: int = TARGET_MONTHS) -> Snapshot:
    feature_start = cutoff - pd.DateOffset(months=feature_months)
    target_end = cutoff + pd.DateOffset(months=target_months)

    in_features = (ledger["InvoiceDate"] >= feature_start) & (ledger["InvoiceDate"] < cutoff)
    in_target = (ledger["InvoiceDate"] >= cutoff) & (ledger["InvoiceDate"] < target_end)

    # Who counts as a customer: bought (a real purchase, not just a return)
    # during the feature window. New customers in the target window are excluded.
    bought = ledger[in_features & ~ledger["IsReturn"]]["Customer ID"].unique()

    feature_rows = ledger[in_features & ledger["Customer ID"].isin(bought)]

    # Target = net money kept in the target window; customers with no rows get 0.
    target = (ledger[in_target & ledger["Customer ID"].isin(bought)]
              .groupby("Customer ID")["TotalPrice"].sum()
              .reindex(bought, fill_value=0.0)
              .rename("target"))
    target.index.name = "Customer ID"

    # Leakage guard: features must end strictly before the cutoff.
    assert feature_rows["InvoiceDate"].max() < cutoff
    return Snapshot(cutoff, feature_start, target_end, feature_rows, target)


def describe(name: str, s: Snapshot) -> None:
    t = s.target
    print(f"\n=== {name} snapshot ===")
    print(f"features : {s.feature_start.date()} -> {(s.cutoff - pd.Timedelta(days=1)).date()}"
          f"   (actual last row: {s.feature_rows['InvoiceDate'].max()})")
    print(f"target   : {s.cutoff.date()} -> {(s.target_end - pd.Timedelta(days=1)).date()}")
    print(f"customers (bought in feature window): {len(t):,}")
    print(f"zero target : {(t == 0).sum():,}  ({(t == 0).mean():.1%})")
    print(f"negative    : {(t < 0).sum():,}  (cancelled more than bought in the target window)")
    print(f"positive    : {(t > 0).sum():,}")
    print(f"target mean {t.mean():,.0f} | median {t.median():,.0f} | 90th pct {t.quantile(.9):,.0f} | max {t.max():,.0f}")


def main() -> None:
    ledger = load_ledger()
    print("Ledger dates:", ledger["InvoiceDate"].min(), "->", ledger["InvoiceDate"].max())

    train = make_snapshot(ledger, TRAIN_CUTOFF)
    test = make_snapshot(ledger, TEST_CUTOFF)
    describe("TRAIN", train)
    describe("TEST", test)

    print("\n=== How the dates line up ===")
    print(f"train target ends (excl.) {train.target_end.date()}  |  test target starts {test.cutoff.date()}"
          f"  -> targets overlap: {train.target_end > test.cutoff}")
    print(f"test target ends (excl.)  {test.target_end.date()}  |  data ends {ledger['InvoiceDate'].max().date()}"
          f"  -> unused tail: {(ledger['InvoiceDate'] >= test.target_end).sum():,} rows")
    both = train.target.index.intersection(test.target.index)
    print(f"customers in BOTH snapshots: {len(both):,} "
          f"({len(both) / len(test.target):.0%} of test customers)")


if __name__ == "__main__":
    main()
