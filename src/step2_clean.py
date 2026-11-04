"""Step 2: clean the raw data into two tables.

  purchases.csv : real sales      (used for RFM, target, most features)
  returns.csv   : cancellations   (kept for a later return-rate feature)

Rules (applied in this order, every dropped row is counted):
  1. Drop rows with no Customer ID          (can't track an unknown customer)
  2. Cancelled invoices (start with 'C')    -> returns table
  3. From the rest, drop Quantity <= 0      (adjustments / damaged goods)
  4. From the rest, drop Price <= 0         (free items, accounting adjustments)
Then add TotalPrice = Quantity * Price.

Run:  python src/step2_clean.py
"""
from pathlib import Path

import pandas as pd

from step1_load_explore import load_raw

ROOT = Path(__file__).resolve().parent.parent
PURCHASES = ROOT / "data" / "purchases.csv"
RETURNS = ROOT / "data" / "returns.csv"


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split raw rows into (purchases, returns, row_accounting)."""
    acct = {"raw rows": len(df)}

    # 1. unknown customers
    df = df[df["Customer ID"].notna()].copy()
    acct["dropped: no Customer ID"] = acct["raw rows"] - len(df)
    df["Customer ID"] = df["Customer ID"].astype(int)

    # 2. cancellations -> returns table
    is_cancel = df["Invoice"].astype(str).str.startswith("C")
    returns = df[is_cancel].copy()
    df = df[~is_cancel]
    acct["moved to returns (C invoices)"] = len(returns)

    # 3 + 4. non-positive quantity / price among the remaining rows
    n = len(df)
    df = df[df["Quantity"] > 0]
    acct["dropped: Quantity <= 0"] = n - len(df)
    n = len(df)
    df = df[df["Price"] > 0]
    acct["dropped: Price <= 0"] = n - len(df)

    purchases = df.copy()
    purchases["TotalPrice"] = purchases["Quantity"] * purchases["Price"]
    returns["TotalPrice"] = returns["Quantity"] * returns["Price"]  # negative

    acct["purchases kept"] = len(purchases)
    total = (acct["dropped: no Customer ID"] + acct["moved to returns (C invoices)"]
             + acct["dropped: Quantity <= 0"] + acct["dropped: Price <= 0"]
             + acct["purchases kept"])
    assert total == acct["raw rows"], "row accounting does not add up"
    return purchases, returns, acct


def main() -> None:
    raw = load_raw()
    purchases, returns, acct = clean(raw)

    print("--- Row accounting ---")
    for k, v in acct.items():
        print(f"{k:34s} {v:>9,}")

    print("\n--- Purchases ---")
    print("customers:", purchases["Customer ID"].nunique(),
          "| invoices:", purchases["Invoice"].nunique())
    print("dates:", purchases["InvoiceDate"].min(), "->", purchases["InvoiceDate"].max())
    print(purchases[["Quantity", "Price", "TotalPrice"]].describe().to_string())

    print("\n--- Returns ---")
    print("customers:", returns["Customer ID"].nunique(),
          "| invoices:", returns["Invoice"].nunique())
    print("returns whose customer never purchased:",
          (~returns["Customer ID"].isin(purchases["Customer ID"])).sum(), "rows")

    print("\n--- Still-suspicious: non-product StockCodes in purchases ---")
    # real product codes start with 5 digits; things like POST, M, DOT, BANK CHARGES don't
    odd = purchases[~purchases["StockCode"].astype(str).str.match(r"^\d{5}")]
    print(odd.groupby("StockCode")["TotalPrice"].agg(["count", "sum"])
          .sort_values("sum", ascending=False).head(12).to_string())

    print("\n--- Still-suspicious: biggest single lines ---")
    cols = ["Invoice", "StockCode", "Description", "Quantity", "Price", "TotalPrice", "Customer ID"]
    print(purchases.nlargest(5, "TotalPrice")[cols].to_string())

    purchases.to_csv(PURCHASES, index=False)
    returns.to_csv(RETURNS, index=False)
    print(f"\nSaved {PURCHASES.name} and {RETURNS.name} in data/")


if __name__ == "__main__":
    main()
