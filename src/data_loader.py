"""
Data Loader module for Online Retail II dataset.
Loads raw data (either from Excel sheets 'Year 2009-2010' and 'Year 2010-2011' or CSV),
concatenates them, and saves to data/processed/raw_combined.parquet.
"""

import os
from pathlib import Path
import pandas as pd


def load_raw_data(data_dir: Path = Path("data")) -> pd.DataFrame:
    raw_dir = data_dir / "raw"
    xlsx_path = raw_dir / "online_retail_II.xlsx"
    csv_path = raw_dir / "online_retail_II.csv"

    if xlsx_path.exists():
        print(f"Loading Excel file from {xlsx_path}...")
        df_2009_2010 = pd.read_excel(xlsx_path, sheet_name="Year 2009-2010")
        print(f"Loaded 'Year 2009-2010' sheet: {df_2009_2010.shape[0]:,} rows")
        df_2010_2011 = pd.read_excel(xlsx_path, sheet_name="Year 2010-2011")
        print(f"Loaded 'Year 2010-2011' sheet: {df_2010_2011.shape[0]:,} rows")
        combined_df = pd.concat([df_2009_2010, df_2010_2011], ignore_index=True)
    elif csv_path.exists():
        print(f"Loading CSV file from {csv_path}...")
        combined_df = pd.read_csv(csv_path, dtype={"Invoice": str, "StockCode": str})
    else:
        raise FileNotFoundError(f"Neither {xlsx_path} nor {csv_path} found in {raw_dir}")

    # Standardize column names if needed
    col_mapping = {
        "Customer ID": "Customer ID",
        "CustomerID": "Customer ID",
        "Price": "Price",
        "UnitPrice": "Price"
    }
    combined_df.rename(columns=col_mapping, inplace=True)
    return combined_df


def save_processed_data(df: pd.DataFrame, data_dir: Path = Path("data")) -> Path:
    proc_dir = data_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = proc_dir / "raw_combined.parquet"

    # Ensure InvoiceDate is datetime for parquet serialization
    if not pd.api.types.is_datetime64_any_dtype(df["InvoiceDate"]):
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # Cast text columns to string to avoid mixed type issues
    for col in ["Invoice", "StockCode", "Description", "Country"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    try:
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
        print(f"Successfully saved {len(df):,} rows to {parquet_path}")
        return parquet_path
    except Exception as e:
        print(f"Parquet export failed ({e}). Falling back to CSV...")
        fallback_path = proc_dir / "raw_combined.csv"
        df.to_csv(fallback_path, index=False)
        print(f"Saved {len(df):,} rows to fallback CSV: {fallback_path}")
        return fallback_path


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Raw combined shape: {df.shape}")
    saved_path = save_processed_data(df)
    file_size_mb = os.path.getsize(saved_path) / (1024 * 1024)
    print(f"Saved file size: {file_size_mb:.2f} MB")
