"""Step 1: load the raw data and LOOK at it. No cleaning yet.

The Excel file has two sheets (Dec 2009-Nov 2010, Dec 2010-Dec 2011).
Reading Excel is slow, so we read once and cache a CSV copy for later steps.

Run:  python src/step1_load_explore.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "data" / "online_retail_II.xlsx"
CACHE = ROOT / "data" / "raw.csv"


def load_raw() -> pd.DataFrame:
    """Read both sheets and stack them. Cached to CSV after the first run."""
    if CACHE.exists():
        return pd.read_csv(CACHE, parse_dates=["InvoiceDate"])
    sheets = pd.read_excel(XLSX, sheet_name=None)  # dict: sheet name -> DataFrame
    print("Sheets found:", {name: len(df) for name, df in sheets.items()})
    df = pd.concat(sheets.values(), ignore_index=True)
    df.to_csv(CACHE, index=False)
    return df


def main() -> None:
    df = load_raw()

    print("\n--- Shape (rows, columns) ---")
    print(df.shape)

    print("\n--- Columns and types ---")
    print(df.dtypes)

    print("\n--- First 5 rows ---")
    print(df.head().to_string())

    print("\n--- Missing values per column ---")
    print(df.isna().sum())

    print("\n--- Date range ---")
    print(df["InvoiceDate"].min(), "->", df["InvoiceDate"].max())

    print("\n--- Numeric summary ---")
    print(df[["Quantity", "Price"]].describe().to_string())

    print("\n--- Things that look suspicious ---")
    inv = df["Invoice"].astype(str)
    print("Cancelled invoices (start with 'C'):", inv.str.startswith("C").sum(), "rows")
    print("Rows with Quantity <= 0:", (df["Quantity"] <= 0).sum())
    print("Rows with Price <= 0:", (df["Price"] <= 0).sum())
    print("Rows with no Customer ID:", df["Customer ID"].isna().sum())
    print("Unique customers:", df["Customer ID"].nunique())
    print("Unique invoices:", df["Invoice"].nunique())


if __name__ == "__main__":
    main()
