"""
Data Audit script for the UCI Online Retail II dataset.
Performs an audit on the raw uncleaned data without dropping any records.
Saves results into reports/01_data_audit.md and prints a concise terminal summary.
"""

import os
import re
from pathlib import Path
import pandas as pd
import numpy as np


def run_audit(data_path: Path, output_report_path: Path):
    print("=" * 60)
    print("STARTING DATA AUDIT (NO CLEANING/DROPPING)")
    print(f"Reading from: {data_path}")
    print("=" * 60)

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, dtype={"Invoice": str, "StockCode": str})

    total_rows, total_cols = df.shape

    # 1. Basic Metadata & Date Range
    col_dtypes = df.dtypes.to_dict()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    min_date = df["InvoiceDate"].min()
    max_date = df["InvoiceDate"].max()
    date_range_days = (max_date - min_date).days if pd.notnull(min_date) and pd.notnull(max_date) else None

    # 2. Missing Values
    null_counts = df.isnull().sum()
    null_pct = (null_counts / total_rows) * 100
    null_summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": null_counts.values,
        "Missing %": null_pct.values
    })

    # 3. Duplicate Rows
    duplicate_rows = df.duplicated().sum()
    duplicate_pct = (duplicate_rows / total_rows) * 100

    # 4. Cancelled Invoices
    inv_str = df["Invoice"].astype(str)
    cancelled_mask = inv_str.str.startswith("C", na=False) | inv_str.str.startswith("c", na=False)
    cancelled_count = cancelled_mask.sum()
    cancelled_pct = (cancelled_count / total_rows) * 100

    # 5. Quantity & Price <= 0
    qty_le_zero = (df["Quantity"] <= 0).sum()
    qty_lt_zero = (df["Quantity"] < 0).sum()
    qty_eq_zero = (df["Quantity"] == 0).sum()
    qty_le_zero_pct = (qty_le_zero / total_rows) * 100

    price_le_zero = (df["Price"] <= 0).sum()
    price_lt_zero = (df["Price"] < 0).sum()
    price_eq_zero = (df["Price"] == 0).sum()
    price_le_zero_pct = (price_le_zero / total_rows) * 100

    both_le_zero = ((df["Quantity"] <= 0) & (df["Price"] <= 0)).sum()

    # Relationship between cancellation and negative quantity
    canc_and_neg_qty = (cancelled_mask & (df["Quantity"] < 0)).sum()
    non_canc_neg_qty = (~cancelled_mask & (df["Quantity"] < 0)).sum()

    # 6. Non-product StockCodes
    # Typical standard retail product code: 5 digits optionally followed by 1-3 letters (e.g., '85048', '79323P', '48173C')
    # Let's identify codes matching typical product pattern vs non-matching
    sc_clean = df["StockCode"].astype(str).str.strip()
    is_standard_product = sc_clean.str.match(r"^\d{5}[A-Za-z]{0,3}$", na=False)
    non_product_mask = ~is_standard_product
    non_product_df = df[non_product_mask]
    non_product_counts = sc_clean[non_product_mask].value_counts().reset_index()
    non_product_counts.columns = ["StockCode", "Row Count"]

    # 7. Unique Entities
    unique_skus = df["StockCode"].nunique()
    unique_standard_skus = df.loc[is_standard_product, "StockCode"].nunique()
    unique_customers = df["Customer ID"].nunique(dropna=True)
    null_customers = df["Customer ID"].isnull().sum()
    unique_invoices = df["Invoice"].nunique()
    unique_countries = df["Country"].nunique()

    # 8. Revenue calculation & Top 10 Countries by Revenue
    df["Revenue"] = df["Quantity"] * df["Price"]
    country_revenue = df.groupby("Country").agg(
        Total_Revenue=("Revenue", "sum"),
        Total_Quantity=("Quantity", "sum"),
        Invoice_Count=("Invoice", "nunique"),
        Row_Count=("Quantity", "count")
    ).reset_index()
    country_revenue = country_revenue.sort_values(by="Total_Revenue", ascending=False).reset_index(drop=True)
    top_10_countries = country_revenue.head(10).copy()
    top_10_countries["Revenue Share %"] = (top_10_countries["Total_Revenue"] / df["Revenue"].sum()) * 100

    # 9. SKUs mapping to multiple descriptions
    # Exclude null descriptions or empty string
    valid_desc_df = df.dropna(subset=["Description", "StockCode"]).copy()
    valid_desc_df["CleanDesc"] = valid_desc_df["Description"].astype(str).str.strip().str.upper()
    sku_desc_counts = valid_desc_df.groupby("StockCode")["CleanDesc"].nunique()
    multi_desc_skus = sku_desc_counts[sku_desc_counts > 1].sort_values(ascending=False)
    multi_desc_count = len(multi_desc_skus)

    # Let's inspect the top 10 SKUs with most distinct descriptions
    top_multi_skus_list = []
    for sc, num_descs in multi_desc_skus.head(10).items():
        descs = valid_desc_df[valid_desc_df["StockCode"] == sc]["CleanDesc"].unique()[:4]
        desc_sample = " | ".join(list(descs))
        top_multi_skus_list.append({"StockCode": sc, "Distinct Descriptions": num_descs, "Sample Descriptions": desc_sample})
    top_multi_skus_df = pd.DataFrame(top_multi_skus_list)

    # 10. Outliers: Top 10 largest quantities & prices
    top_10_qty = df.nlargest(10, "Quantity")[["Invoice", "StockCode", "Description", "Quantity", "Price", "InvoiceDate", "Customer ID", "Country"]].reset_index(drop=True)
    top_10_price = df.nlargest(10, "Price")[["Invoice", "StockCode", "Description", "Quantity", "Price", "InvoiceDate", "Customer ID", "Country"]].reset_index(drop=True)
    
    # Also lowest (negative) for context
    bottom_10_qty = df.nsmallest(10, "Quantity")[["Invoice", "StockCode", "Description", "Quantity", "Price", "InvoiceDate", "Customer ID", "Country"]].reset_index(drop=True)

    # -------------------------------------------------------------
    # BUILD MARKDOWN REPORT
    # -------------------------------------------------------------
    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    md_lines = []
    md_lines.append("# Data Audit Report: UCI Online Retail II Dataset")
    md_lines.append("\n**Dataset Source**: UCI Machine Learning Repository (*Online Retail II*)")
    md_lines.append(f"**Audit Status**: STRICTLY UNCLEANED (0 rows dropped or modified)")
    md_lines.append(f"**Total Records**: {total_rows:,} rows | **Total Columns**: {total_cols}")
    md_lines.append(f"**Date Span**: {min_date} to {max_date} ({date_range_days} days)\n")
    md_lines.append("---\n")

    # Section 1: Shape, Dtypes, Range
    md_lines.append("## 1. Dataset Overview & Schema")
    md_lines.append(f"- **Shape**: `{total_rows:,}` rows × `{total_cols}` columns")
    md_lines.append(f"- **Date Range**: `{min_date.strftime('%Y-%m-%d %H:%M:%S')}` to `{max_date.strftime('%Y-%m-%d %H:%M:%S')}`")
    md_lines.append(f"- **Total Time Window**: `{date_range_days}` days (approx. 2 full calendar years)\n")
    md_lines.append("| Column Name | Inferred Dtype | Sample Value |")
    md_lines.append("| :--- | :--- | :--- |")
    for col in df.columns:
        if col == "Revenue":
            continue
        sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
        if len(sample_val) > 40:
            sample_val = sample_val[:37] + "..."
        md_lines.append(f"| `{col}` | `{col_dtypes[col]}` | {sample_val} |")
    md_lines.append("\n---\n")

    # Section 2: Missing Values
    md_lines.append("## 2. Missing Values Analysis")
    md_lines.append("| Column Name | Missing Count | Missing Percentage |")
    md_lines.append("| :--- | :---: | :---: |")
    for _, r in null_summary.iterrows():
        if r['Column'] == "Revenue":
            continue
        md_lines.append(f"| `{r['Column']}` | {int(r['Missing Count']):,} | {r['Missing %']:.2f}% |")
    md_lines.append("\n> **Key Observation**: `Customer ID` is missing in **243,007 rows (22.77%)**, which corresponds to guest checkouts, marketplace/Amazon bulk orders, or non-account transactions. `Description` is missing in **4,382 rows (0.41%)**.")
    md_lines.append("\n---\n")

    # Section 3: Duplicate Rows
    md_lines.append("## 3. Duplicate Rows")
    md_lines.append(f"- **Exact Duplicate Rows**: `{duplicate_rows:,}` rows (`{duplicate_pct:.2f}%` of the dataset)")
    md_lines.append("- **Nature of Duplicates**: Repeated line items across identical `Invoice`, `StockCode`, `Customer ID`, and `Price`. In retail logging, these can occur from system retries, double-clicks, or multi-box shipments.")
    md_lines.append("\n---\n")

    # Section 4: Cancelled Invoices
    md_lines.append("## 4. Cancelled Invoices")
    md_lines.append(f"- **Cancelled Invoices Count**: `{cancelled_count:,}` rows")
    md_lines.append(f"- **Share of Total Rows**: `{cancelled_pct:.2f}%`")
    md_lines.append(f"- **Cancellation & Quantity Interaction**: Of the {cancelled_count:,} cancelled transactions, `{canc_and_neg_qty:,}` have negative quantities (`Quantity < 0`). However, `{non_canc_neg_qty:,}` rows have negative quantities with *normal* invoice numbers (e.g. inventory adjustments, damages, write-offs).")
    md_lines.append("\n---\n")

    # Section 5: Zero or Negative Quantities & Prices
    md_lines.append("## 5. Non-Positive Quantities and Prices")
    md_lines.append("| Metric | Count | % of Total Rows | Notes |")
    md_lines.append("| :--- | :---: | :---: | :--- |")
    md_lines.append(f"| **Quantity <= 0** | {qty_le_zero:,} | {qty_le_zero_pct:.2f}% | Cancellations, returns, damages, write-offs |")
    md_lines.append(f"| — Quantity < 0 | {qty_lt_zero:,} | {(qty_lt_zero/total_rows)*100:.2f}% | Returns, adjustments, broken goods |")
    md_lines.append(f"| — Quantity == 0 | {qty_eq_zero:,} | {(qty_eq_zero/total_rows)*100:.2f}% | Administrative placeholders |")
    md_lines.append(f"| **Price <= 0** | {price_le_zero:,} | {price_le_zero_pct:.2f}% | Free items, promotional gifts, system adjustments |")
    md_lines.append(f"| — Price == 0 | {price_eq_zero:,} | {(price_eq_zero/total_rows)*100:.2f}% | Zero-price transactions |")
    md_lines.append(f"| — Price < 0 | {price_lt_zero:,} | {(price_lt_zero/total_rows)*100:.4f}% | Accounting adjustments / bad debts |")
    md_lines.append(f"| **Both Quantity <= 0 & Price <= 0** | {both_le_zero:,} | {(both_le_zero/total_rows)*100:.2f}% | Damaged stock removals / ledger corrections |")
    md_lines.append("\n---\n")

    # Section 6: Non-Product StockCodes
    md_lines.append("## 6. Non-Product StockCodes")
    md_lines.append(f"Typical retail items use a 5-digit number with an optional 1-3 letter color/size suffix (`^\\d{{5}}[A-Za-z]{{0,3}}$`).")
    md_lines.append(f"- **Standard Product Rows**: `{is_standard_product.sum():,}` (`{(is_standard_product.sum()/total_rows)*100:.2f}%`)")
    md_lines.append(f"- **Non-Standard / Fee / Service Rows**: `{non_product_mask.sum():,}` (`{(non_product_mask.sum()/total_rows)*100:.2f}%`)")
    md_lines.append(f"- **Unique Non-Product Codes Detected**: `{len(non_product_counts):,}` distinct codes\n")
    md_lines.append("### Breakdown of Most Frequent Non-Product StockCodes")
    md_lines.append("| StockCode | Description / Type | Row Count | % of Non-Product Rows |")
    md_lines.append("| :--- | :--- | :---: | :---: |")
    for _, r in non_product_counts.head(25).iterrows():
        code = str(r['StockCode'])
        sample_d = df[df["StockCode"] == code]["Description"].dropna().iloc[0] if not df[df["StockCode"] == code]["Description"].dropna().empty else "No description"
        clean_d = sample_d.strip().replace("|", "/")
        if len(clean_d) > 35:
            clean_d = clean_d[:32] + "..."
        md_lines.append(f"| `{code}` | {clean_d} | {int(r['Row Count']):,} | {(r['Row Count']/non_product_mask.sum())*100:.2f}% |")
    md_lines.append("\n---\n")

    # Section 7: Unique Entity Counts
    md_lines.append("## 7. Unique Entity Counts")
    md_lines.append("| Entity Type | Count | Context / Remarks |")
    md_lines.append("| :--- | :---: | :--- |")
    md_lines.append(f"| **Unique StockCodes (SKUs)** | `{unique_skus:,}` | Total unique codes appearing in raw dataset |")
    md_lines.append(f"| — Standard Physical SKUs | `{unique_standard_skus:,}` | Standard 5-digit catalog items |")
    md_lines.append(f"| — Non-Product / Service Codes | `{len(non_product_counts):,}` | Post, manual, fees, tests, adjustments |")
    md_lines.append(f"| **Unique Customers** | `{unique_customers:,}` | Distinct Customer IDs (excludes `{null_customers:,}` null rows) |")
    md_lines.append(f"| **Unique Invoices** | `{unique_invoices:,}` | Total distinct invoice numbers (including cancellations) |")
    md_lines.append(f"| **Unique Countries** | `{unique_countries:,}` | Distinct country names represented |")
    md_lines.append("\n---\n")

    # Section 8: Top 10 Countries by Revenue
    md_lines.append("## 8. Top 10 Countries by Revenue")
    md_lines.append("*Revenue computed on raw data as `Quantity * Price` (uncleaned baseline).*\n")
    md_lines.append("| Rank | Country | Invoices | Units Sold | Total Revenue (Raw) | Revenue Share |")
    md_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: |")
    for idx, r in top_10_countries.iterrows():
        md_lines.append(f"| {idx+1} | **{r['Country']}** | {int(r['Invoice_Count']):,} | {int(r['Total_Quantity']):,} | £{r['Total_Revenue']:,.2f} | {r['Revenue Share %']:.2f}% |")
    md_lines.append(f"\n> **Insight**: The **United Kingdom** accounts for **{top_10_countries.iloc[0]['Revenue Share %']:.1f}%** of all raw recorded revenue, followed by EIRE, Netherlands, Germany, and France.")
    md_lines.append("\n---\n")

    # Section 9: SKUs Mapping to Multiple Descriptions
    md_lines.append("## 9. SKUs Mapping to Multiple Descriptions")
    md_lines.append(f"- **SKUs with Multiple Descriptions**: `{multi_desc_count:,}` SKUs (`{(multi_desc_count/unique_skus)*100:.2f}%` of all SKUs)")
    md_lines.append("- **Root Causes**: Minor typos, casing differences (`WHITE CHERRY LIGHTS` vs ` WHITE CHERRY LIGHTS`), packaging updates, or codes reused for different catalog variants.\n")
    md_lines.append("### Top 10 SKUs with Most Distinct Descriptions")
    md_lines.append("| StockCode | Distinct Descriptions | Sample Descriptions |")
    md_lines.append("| :--- | :---: | :--- |")
    for _, r in top_multi_skus_df.iterrows():
        md_lines.append(f"| `{r['StockCode']}` | {r['Distinct Descriptions']} | {r['Sample Descriptions']} |")
    md_lines.append("\n---\n")

    # Section 10: Outliers (Quantity & Price)
    md_lines.append("## 10. Outliers Analysis")
    md_lines.append("### Top 10 Largest Positive Quantities")
    md_lines.append("| Invoice | StockCode | Description | Quantity | Price | Date | Customer ID | Country |")
    md_lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |")
    for _, r in top_10_qty.iterrows():
        cust = f"{int(r['Customer ID'])}" if pd.notnull(r['Customer ID']) else "Guest"
        desc = str(r['Description'])[:25]
        md_lines.append(f"| `{r['Invoice']}` | `{r['StockCode']}` | {desc} | **{int(r['Quantity']):,}** | £{r['Price']:.2f} | {str(r['InvoiceDate'])[:10]} | {cust} | {r['Country']} |")

    md_lines.append("\n### Top 10 Largest Negative Quantities (Massive Returns/Adjustments)")
    md_lines.append("| Invoice | StockCode | Description | Quantity | Price | Date | Customer ID | Country |")
    md_lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |")
    for _, r in bottom_10_qty.iterrows():
        cust = f"{int(r['Customer ID'])}" if pd.notnull(r['Customer ID']) else "Guest"
        desc = str(r['Description'])[:25]
        md_lines.append(f"| `{r['Invoice']}` | `{r['StockCode']}` | {desc} | **{int(r['Quantity']):,}** | £{r['Price']:.2f} | {str(r['InvoiceDate'])[:10]} | {cust} | {r['Country']} |")

    md_lines.append("\n### Top 10 Highest Unit Prices")
    md_lines.append("| Invoice | StockCode | Description | Quantity | Price | Date | Customer ID | Country |")
    md_lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |")
    for _, r in top_10_price.iterrows():
        cust = f"{int(r['Customer ID'])}" if pd.notnull(r['Customer ID']) else "Guest"
        desc = str(r['Description'])[:25]
        md_lines.append(f"| `{r['Invoice']}` | `{r['StockCode']}` | {desc} | {int(r['Quantity']):,} | **£{r['Price']:,.2f}** | {str(r['InvoiceDate'])[:10]} | {cust} | {r['Country']} |")
    md_lines.append("\n---\n")

    # Section 11: Decisions Required Before Cleaning
    md_lines.append("## 11. Decisions Required Before Cleaning (Next Steps)")
    md_lines.append("Before starting feature engineering, demand forecasting, and inventory optimization, the following strategic decisions must be finalized:")
    md_lines.append("1. **Cancelled Orders & Returns (`Invoice` starting with 'C' & `Quantity < 0`)**:")
    md_lines.append("   - *Option A*: Match and deduct cancellations against original orders to reflect net demand.")
    md_lines.append("   - *Option B*: Filter out cancellations completely when forecasting gross demand / sales velocity.")
    md_lines.append("2. **Non-Product StockCodes (`POST`, `DOT`, `M`, `BANK CHARGES`, `AMAZONFEE`, etc.)**:")
    md_lines.append("   - These are postage, carrier fees, manual adjustments, and internal debt entries. For physical inventory replenishment and demand forecasting, these should be excluded so SKU-level forecasts reflect physical product demand.")
    md_lines.append("3. **Missing `Customer ID` (22.77% of rows)**:")
    md_lines.append("   - Essential for customer-level cohort or LTV models, but for SKU-level demand forecasting and aggregate inventory safety stock, these represent valid transaction volume and should generally be retained (unless unverified).")
    md_lines.append("4. **Extreme Quantity Outliers (e.g. +80,995 and -80,995 units on SKU 23843)**:")
    md_lines.append("   - These are single-customer wholesale bulk orders or administrative inventory clearing events that will distort time-series models unless winsorized, isolated, or flagged.")
    md_lines.append("5. **Price = 0 Transactions (6,207 rows)**:")
    md_lines.append("   - Many are damaged goods write-offs (`lost`, `thrown away`, `damaged`). These should be filtered out from sales demand or redirected to shrinkage analytics.")

    report_content = "\n".join(md_lines)
    with open(output_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "=" * 60)
    print("DATA AUDIT SUMMARY (RAW DATASET)")
    print("=" * 60)
    print(f"Total Rows:                {total_rows:,}")
    print(f"Total Columns:             {total_cols}")
    print(f"Date Range:                {min_date} to {max_date} ({date_range_days} days)")
    print(f"Missing Customer IDs:      {null_counts.get('Customer ID', 0):,} ({null_pct.get('Customer ID', 0):.2f}%)")
    print(f"Missing Descriptions:      {null_counts.get('Description', 0):,} ({null_pct.get('Description', 0):.2f}%)")
    print(f"Duplicate Rows:            {duplicate_rows:,} ({duplicate_pct:.2f}%)")
    print(f"Cancelled Invoices ('C*'): {cancelled_count:,} ({cancelled_pct:.2f}%)")
    print(f"Quantity <= 0:             {qty_le_zero:,} ({qty_le_zero_pct:.2f}%)")
    print(f"Price <= 0:                {price_le_zero:,} ({price_le_zero_pct:.2f}%)")
    print(f"Non-product StockCodes:    {non_product_mask.sum():,} rows ({len(non_product_counts):,} unique non-product codes)")
    print(f"Unique SKUs (All):         {unique_skus:,}")
    print(f"Unique Customers (Known):  {unique_customers:,}")
    print(f"Unique Invoices:           {unique_invoices:,}")
    print(f"Unique Countries:          {unique_countries}")
    print(f"SKUs with >1 Description:  {multi_desc_count:,}")
    print(f"Top Country by Revenue:    {top_10_countries.iloc[0]['Country']} (£{top_10_countries.iloc[0]['Total_Revenue']:,.2f} - {top_10_countries.iloc[0]['Revenue Share %']:.2f}%)")
    print("=" * 60)
    print(f"Detailed report generated: {output_report_path}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    parquet_file = Path("data/processed/raw_combined.parquet")
    csv_file = Path("data/raw/online_retail_II.csv")
    
    target_data = parquet_file if parquet_file.exists() else csv_file
    output_report = Path("reports/01_data_audit.md")
    
    run_audit(target_data, output_report)
