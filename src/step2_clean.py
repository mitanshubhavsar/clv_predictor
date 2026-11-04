"""Step 2: clean the raw data into three tables.

  purchases.csv : real sales                  (counts: Frequency, AOV, ...)
  returns.csv   : cancellations (negative $)  (for the return-rate feature)
  ledger.csv    : purchases + returns, one table of signed money
                  (for Monetary and the 6-month target = money KEPT)

Decisions:
  * Returns are NETTED off spend. A fully cancelled order is worth 0, not its
    gross value. Done by stacking returns as negative rows dated when the
    cancellation happened, so spend in any window = sum of that window's rows.
  * Internal / non-customer stock codes are dropped (manual entries,
    adjustments, bank charges, tests). Postage / carriage (POST, C2, DOT) is
    KEPT: customers really pay it.

Rules (applied in this order, every dropped row is counted):
  1. Drop rows with no Customer ID          (can't track an unknown customer)
  2. Drop internal stock codes              (not customer behaviour)
  3. Cancelled invoices (start with 'C')    -> returns table
  4. From the rest, drop Quantity <= 0
  5. From the rest, drop Price <= 0
Then add TotalPrice = Quantity * Price (negative for returns).

Run:  python src/step2_clean.py
"""
from pathlib import Path

import pandas as pd

from step1_load_explore import load_raw

ROOT = Path(__file__).resolve().parent.parent
PURCHASES = ROOT / "data" / "purchases.csv"
RETURNS = ROOT / "data" / "returns.csv"
LEDGER = ROOT / "data" / "ledger.csv"

# Not customer behaviour: manual entries, adjustments, fees, tests, samples.
INTERNAL_CODES = {
    "M", "ADJUST", "ADJUST2", "BANK CHARGES", "TEST001", "TEST002",
    "D", "SP1002", "PADS",
}
# Kept on purpose (postage / carriage paid by customers): POST, C2, DOT


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split raw rows into (purchases, returns, row_accounting)."""
    acct = {"raw rows": len(df)}

    # 1. unknown customers
    df = df[df["Customer ID"].notna()].copy()
    acct["dropped: no Customer ID"] = acct["raw rows"] - len(df)
    df["Customer ID"] = df["Customer ID"].astype(int)

    # 2. internal stock codes (from purchases AND cancellations)
    n = len(df)
    df = df[~df["StockCode"].astype(str).isin(INTERNAL_CODES)]
    acct["dropped: internal stock codes"] = n - len(df)

    # 3. cancellations -> returns table
    is_cancel = df["Invoice"].astype(str).str.startswith("C")
    returns = df[is_cancel].copy()
    df = df[~is_cancel]
    acct["moved to returns (C invoices)"] = len(returns)

    # 4 + 5. non-positive quantity / price among the remaining rows
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
    total = (acct["dropped: no Customer ID"] + acct["dropped: internal stock codes"]
             + acct["moved to returns (C invoices)"] + acct["dropped: Quantity <= 0"]
             + acct["dropped: Price <= 0"] + acct["purchases kept"])
    assert total == acct["raw rows"], "row accounting does not add up"
    return purchases, returns, acct


def build_ledger(purchases: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Stack purchases and returns into one signed-money table.

    Returns already have negative TotalPrice, dated at the cancellation, so
    summing TotalPrice over any date window gives money kept in that window.
    """
    ledger = pd.concat(
        [purchases.assign(IsReturn=False), returns.assign(IsReturn=True)],
        ignore_index=True,
    )
    return ledger.sort_values(["Customer ID", "InvoiceDate"]).reset_index(drop=True)


def main() -> None:
    raw = load_raw()
    purchases, returns, acct = clean(raw)
    ledger = build_ledger(purchases, returns)

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

    print("\n--- Gross vs net spend (the effect of netting) ---")
    gross = purchases["TotalPrice"].sum()
    net = ledger["TotalPrice"].sum()
    print(f"gross purchases: {gross:>14,.0f}")
    print(f"returns:         {returns['TotalPrice'].sum():>14,.0f}")
    print(f"net (ledger):    {net:>14,.0f}")
    per_cust = ledger.groupby("Customer ID")["TotalPrice"].sum()
    print("customers with NEGATIVE net spend overall:", (per_cust < 0).sum())

    print("\n--- Remaining non-product codes in purchases (should be postage only) ---")
    odd = purchases[~purchases["StockCode"].astype(str).str.match(r"^\d{5}")]
    print(odd.groupby("StockCode")["TotalPrice"].agg(["count", "sum"])
          .sort_values("sum", ascending=False).head(8).to_string())

    purchases.to_csv(PURCHASES, index=False)
    returns.to_csv(RETURNS, index=False)
    ledger.to_csv(LEDGER, index=False)
    print(f"\nSaved {PURCHASES.name}, {RETURNS.name}, {LEDGER.name} in data/")


if __name__ == "__main__":
    main()
