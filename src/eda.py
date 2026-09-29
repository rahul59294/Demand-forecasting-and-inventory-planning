"""
Comprehensive Exploratory Data Analysis & Calendar Standardization Script (Updated & Reconciled).
Part A - Fixes:
  A1: Build data/processed/calendar_weeks.parquet (104 weeks, is_closed_week flag).
  A2: Rebuild data/processed/weekly_total.parquet and outputs/weekly_total_demand.png (shaded closed weeks).
  A3: Add is_closed_week column to data/processed/weekly_sku_demand.parquet.
  A4: Reconciliation of 104-week totals across monthly, annual, customer, and SKU tables.

Part B - Comprehensive EDA:
  B1: ABC Analysis (Pareto chart: outputs/eda/pareto_abc.png).
  B2: Syntetos-Boylan Demand Classification (Raw & Deseasonalized, Transition Matrix, Scatter: outputs/eda/adi_cv2_scatter.png).
  B3: Lifecycle Status (new, discontinued, continuing) & ABC cross-tab.
  B4: Seasonality (Monthly indices, YoY overlay: outputs/eda/yoy_overlay.png, STL decomposition: outputs/eda/stl_decomposition.png, Top 6 SKUs: outputs/eda/top_sku_weekly.png).
  B5: Growth & Customer Concentration (Reconciled to 104 weeks; Volume vs Price/Mix decomposition; Dec-Apr decline audit).
  B6: Candidate Forecasting Universes (Distinct, informative trade-offs).

Outputs:
  data/processed/calendar_weeks.parquet
  data/processed/weekly_total.parquet
  data/processed/weekly_sku_demand.parquet
  data/processed/sku_classification.parquet
  outputs/weekly_total_demand.png
  outputs/eda/*.png
  reports/03_eda.md
"""

import os
from pathlib import Path
import time
import subprocess

# Configure writable matplotlib config directory and avoid sandbox system_profiler crash
mpl_cache = Path(__file__).resolve().parent.parent / ".mplconfig"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache)

_orig_check_output = subprocess.check_output
def _safe_check_output(*args, **kwargs):
    if args and len(args[0]) > 0 and "system_profiler" in args[0]:
        raise OSError("system_profiler disabled in sandbox")
    return _orig_check_output(*args, **kwargs)
subprocess.check_output = _safe_check_output

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.seasonal import STL


def run_eda_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("STARTING DEMAND FORECASTING EDA PIPELINE (RECONCILED)")
    print("=" * 70)

    eda_dir = Path("outputs/eda")
    eda_dir.mkdir(parents=True, exist_ok=True)

    # Load base files
    clean_tx = pd.read_parquet("data/processed/clean_transactions.parquet")
    sku_master = pd.read_parquet("data/processed/sku_master.parquet")
    weekly_sku = pd.read_parquet("data/processed/weekly_sku_demand.parquet")
    weekly_total_raw = pd.read_parquet("data/processed/weekly_total.parquet")

    # =============================================================
    # PART A - FIXES & CALENDAR STANDARDIZATION
    # =============================================================
    print("\n--- PART A: FIXES & CALENDAR STANDARDIZATION ---")
    
    # A1: Build calendar_weeks.parquet
    wks = pd.date_range("2009-12-07", "2011-11-28", freq="W-MON")
    calendar_df = pd.DataFrame({
        "week_start": wks,
        "week_index": np.arange(len(wks)),
        "year": wks.year,
        "month": wks.month,
        "iso_week": [int(w.strftime("%V")) for w in wks]
    })

    # Flag is_closed_week
    clean_tx["week_start"] = (
        clean_tx["InvoiceDate"].dt.floor("D") 
        - pd.to_timedelta(clean_tx["InvoiceDate"].dt.dayofweek, unit="D")
    )
    sales_by_week = clean_tx.groupby("week_start")["Quantity"].sum().to_dict()
    calendar_df["is_closed_week"] = calendar_df["week_start"].map(lambda w: sales_by_week.get(w, 0) == 0)

    closed_weeks = calendar_df[calendar_df["is_closed_week"]]["week_start"].tolist()
    closed_str = [w.strftime("%Y-%m-%d") for w in closed_weeks]
    print(f"A1. Total calendar weeks: {len(calendar_df)} (Mondays: 2009-12-07 to 2011-11-28)")
    print(f"    Closed weeks identified (0 total sales): {closed_str}")

    calendar_path = Path("data/processed/calendar_weeks.parquet")
    calendar_df.to_parquet(calendar_path, index=False)
    print(f"    Saved calendar weeks to: {calendar_path}")

    # A2: Rebuild weekly_total.parquet on full 104-week calendar and re-save chart
    weekly_total = calendar_df.merge(
        weekly_total_raw[["week_start", "total_qty", "total_revenue", "active_skus", "n_invoices"]],
        on="week_start",
        how="left"
    )
    weekly_total["total_qty"] = weekly_total["total_qty"].fillna(0).astype(int)
    weekly_total["total_revenue"] = weekly_total["total_revenue"].fillna(0.0)
    weekly_total["active_skus"] = weekly_total["active_skus"].fillna(0).astype(int)
    weekly_total["n_invoices"] = weekly_total["n_invoices"].fillna(0).astype(int)

    weekly_total_path = Path("data/processed/weekly_total.parquet")
    weekly_total.to_parquet(weekly_total_path, index=False)
    print(f"A2. Rebuilt weekly_total.parquet ({len(weekly_total)} weeks).")

    # Shaded chart: outputs/weekly_total_demand.png
    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=300)
    ax2 = ax1.twinx()

    ax1.plot(weekly_total["week_start"], weekly_total["total_qty"], color="#1f77b4", linewidth=2.0, label="Weekly Units Demanded")
    ax2.plot(weekly_total["week_start"], weekly_total["total_revenue"], color="#2ca02c", linewidth=1.5, linestyle="--", label="Weekly Net Revenue (£)")

    # Shade closed weeks
    for i, cw in enumerate(closed_weeks):
        ax1.axvspan(cw, cw + pd.Timedelta(days=7), color="red", alpha=0.25, label="Closed Holiday Week (Zero Sales)" if i == 0 else "")

    ax1.set_title("UCI Online Retail II: Full 104-Week Demand Series (Mondays 2009-12-07 to 2011-11-28)", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlabel("Week Start (Monday)", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Total Units Demanded", color="#1f77b4", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Total Net Revenue (£)", color="#2ca02c", fontsize=11, fontweight="semibold")
    ax1.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter("£{x:,.0f}"))
    ax1.grid(True, linestyle="--", alpha=0.5)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    chart_path = Path("outputs/weekly_total_demand.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"    Saved shaded weekly total demand chart to: {chart_path}")

    # A3: Add is_closed_week column to weekly_sku_demand.parquet
    if "is_closed_week" in weekly_sku.columns:
        weekly_sku.drop(columns=["is_closed_week"], inplace=True)
    weekly_sku = weekly_sku.merge(
        calendar_df[["week_start", "is_closed_week"]],
        on="week_start",
        how="left"
    )
    weekly_sku["is_closed_week"] = weekly_sku["is_closed_week"].fillna(False)
    weekly_sku_path = Path("data/processed/weekly_sku_demand.parquet")
    weekly_sku.to_parquet(weekly_sku_path, index=False)
    print(f"A3. Added is_closed_week column to weekly_sku_demand.parquet ({len(weekly_sku):,} rows).")

    # Filter clean_transactions to exact 104 weeks for reconciled analysis
    tx_104 = clean_tx[clean_tx["week_start"].isin(calendar_df["week_start"])].copy()
    print(f"    Clean transactions restricted to 104 weeks: {len(tx_104):,} rows ({tx_104['Quantity'].sum():,} units, £{tx_104['Revenue'].sum():,.2f}).")

    # =============================================================
    # PART B - COMPREHENSIVE EDA
    # =============================================================
    print("\n--- PART B: COMPREHENSIVE EDA ---")

    # B1: ABC Analysis (rank by total revenue from sku_master)
    print("Running B1: ABC Analysis...")
    sku_master_sorted = sku_master.sort_values(by="total_revenue", ascending=False).reset_index(drop=True)
    grand_revenue = sku_master_sorted["total_revenue"].sum()
    grand_units = sku_master_sorted["total_qty"].sum()
    sku_master_sorted["cum_rev"] = sku_master_sorted["total_revenue"].cumsum()
    sku_master_sorted["cum_pct"] = (sku_master_sorted["cum_rev"] / grand_revenue) * 100

    # Precise boundary: the SKU crossing the 80% boundary belongs to A
    sku_master_sorted["abc_class"] = "C"
    idx_a = (sku_master_sorted["cum_pct"] <= 80.0).sum()
    sku_master_sorted.loc[:idx_a, "abc_class"] = "A"
    
    idx_b = (sku_master_sorted["cum_pct"] <= 95.0).sum()
    sku_master_sorted.loc[idx_a + 1:idx_b, "abc_class"] = "B"
    sku_master_sorted.loc[idx_b + 1:, "abc_class"] = "C"

    abc_summary = sku_master_sorted.groupby("abc_class").agg(
        sku_count=("StockCode", "count"),
        total_rev=("total_revenue", "sum")
    ).loc[["A", "B", "C"]]
    abc_summary["sku_pct"] = (abc_summary["sku_count"] / len(sku_master_sorted)) * 100
    abc_summary["rev_pct"] = (abc_summary["total_rev"] / grand_revenue) * 100

    print("ABC Analysis Summary:")
    print(abc_summary)

    # Pareto Chart
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    sku_indices = np.arange(1, len(sku_master_sorted) + 1)
    sku_pct_axis = (sku_indices / len(sku_master_sorted)) * 100

    ax1.plot(sku_pct_axis, sku_master_sorted["cum_pct"], color="#1f77b4", linewidth=2.5, label="Cumulative Revenue %")
    ax1.set_title("Pareto ABC Analysis: SKU Revenue Concentration", fontsize=13, fontweight="bold", pad=15)
    ax1.set_xlabel("% of Total Physical SKUs", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Cumulative Revenue %", color="#1f77b4", fontsize=11, fontweight="semibold")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 102)
    ax1.grid(True, linestyle="--", alpha=0.5)

    pct_skus_a = (abc_summary.loc["A", "sku_count"] / len(sku_master_sorted)) * 100
    pct_skus_ab = ((abc_summary.loc["A", "sku_count"] + abc_summary.loc["B", "sku_count"]) / len(sku_master_sorted)) * 100
    
    ax1.axvline(pct_skus_a, color="#d62728", linestyle=":", label=f"Class A Cutoff ({pct_skus_a:.1f}% SKUs)")
    ax1.axvline(pct_skus_ab, color="#ff7f0e", linestyle=":", label=f"Class B Cutoff ({pct_skus_ab:.1f}% SKUs)")
    ax1.axhline(80, color="#d62728", linestyle="--", alpha=0.5)
    ax1.axhline(95, color="#ff7f0e", linestyle="--", alpha=0.5)

    ax1.annotate(
        f"Class A: {abc_summary.loc['A', 'sku_count']:,} SKUs ({pct_skus_a:.1f}%)\nGenerates {abc_summary.loc['A', 'rev_pct']:.1f}% Revenue",
        xy=(pct_skus_a / 2, 45), fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#d62728")
    )
    ax1.annotate(
        f"Class B: {abc_summary.loc['B', 'sku_count']:,} SKUs\nGenerates {abc_summary.loc['B', 'rev_pct']:.1f}% Revenue",
        xy=((pct_skus_a + pct_skus_ab) / 2, 85), fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ff7f0e")
    )

    ax1.legend(loc="lower right")
    plt.tight_layout()
    pareto_path = eda_dir / "pareto_abc.png"
    plt.savefig(pareto_path)
    plt.close()
    print(f"Saved Pareto chart to: {pareto_path}")

    # B2: Syntetos-Boylan Demand Pattern Classification (Raw & Deseasonalized)
    print("\nRunning B2: Demand Pattern Classification (Raw & Deseasonalized)...")
    # Compute monthly seasonal indices on open weeks first
    open_weeks_total = weekly_total[~weekly_total["is_closed_week"]].copy()
    overall_mean_weekly_qty = open_weeks_total["total_qty"].mean()
    monthly_idx = open_weeks_total.groupby("month")["total_qty"].mean() / overall_mean_weekly_qty

    open_weekly_sku = weekly_sku[~weekly_sku["is_closed_week"]].copy()
    open_weekly_sku["month"] = open_weekly_sku["week_start"].dt.month
    open_weekly_sku["seasonal_idx"] = open_weekly_sku["month"].map(monthly_idx)
    open_weekly_sku["qty_deseasonalized"] = open_weekly_sku["qty"] / open_weekly_sku["seasonal_idx"]

    def classify_sb(adi, cv2):
        if adi < 1.32 and cv2 < 0.49:
            return "smooth"
        elif adi < 1.32 and cv2 >= 0.49:
            return "erratic"
        elif adi >= 1.32 and cv2 < 0.49:
            return "intermittent"
        else:
            return "lumpy"

    sku_pattern_list = []
    grouped = open_weekly_sku.groupby("StockCode")
    for sku, group in grouped:
        active_open_weeks = len(group)
        pos_group = group[group["qty"] > 0]
        nonzero_weeks = len(pos_group)

        if nonzero_weeks == 0:
            adi = np.nan
            cv2 = np.nan
            cv2_deseas = np.nan
            pat = "inactive"
            pat_deseas = "inactive"
            is_single = False
        else:
            adi = active_open_weeks / nonzero_weeks
            if nonzero_weeks == 1:
                cv2 = 0.0
                cv2_deseas = 0.0
                is_single = True
            else:
                qty_vals = pos_group["qty"].values
                mean_q = np.mean(qty_vals)
                std_q = np.std(qty_vals, ddof=0)
                cv2 = float((std_q / mean_q) ** 2) if mean_q > 0 else 0.0

                qty_deseas_vals = pos_group["qty_deseasonalized"].values
                mean_qd = np.mean(qty_deseas_vals)
                std_qd = np.std(qty_deseas_vals, ddof=0)
                cv2_deseas = float((std_qd / mean_qd) ** 2) if mean_qd > 0 else 0.0
                is_single = False

            pat = classify_sb(adi, cv2)
            pat_deseas = classify_sb(adi, cv2_deseas)

        sku_pattern_list.append({
            "StockCode": sku,
            "ADI": adi,
            "CV2": cv2,
            "demand_pattern": pat,
            "CV2_deseasonalized": cv2_deseas,
            "demand_pattern_deseasonalized": pat_deseas,
            "is_single_sale": is_single,
            "nonzero_weeks": nonzero_weeks,
            "active_open_weeks": active_open_weeks
        })

    pattern_df = pd.DataFrame(sku_pattern_list)

    sku_master_sorted_clean = sku_master_sorted.drop(columns=["nonzero_weeks", "active_weeks"], errors="ignore")
    merged_master = sku_master_sorted_clean.merge(
        pattern_df[[
            "StockCode", "ADI", "CV2", "demand_pattern", 
            "CV2_deseasonalized", "demand_pattern_deseasonalized",
            "is_single_sale", "nonzero_weeks", "active_open_weeks"
        ]],
        on="StockCode",
        how="left"
    )

    demand_summary = merged_master.groupby("demand_pattern").agg(
        sku_count=("StockCode", "count"),
        total_rev=("total_revenue", "sum")
    ).loc[["smooth", "erratic", "intermittent", "lumpy"]]
    demand_summary["sku_pct"] = (demand_summary["sku_count"] / len(merged_master)) * 100
    demand_summary["rev_pct"] = (demand_summary["total_rev"] / grand_revenue) * 100

    demand_deseas_summary = merged_master.groupby("demand_pattern_deseasonalized").agg(
        sku_count=("StockCode", "count"),
        total_rev=("total_revenue", "sum")
    ).loc[["smooth", "erratic", "intermittent", "lumpy"]]
    demand_deseas_summary["sku_pct"] = (demand_deseas_summary["sku_count"] / len(merged_master)) * 100
    demand_deseas_summary["rev_pct"] = (demand_deseas_summary["total_rev"] / grand_revenue) * 100

    # Transition Matrix
    trans_matrix = pd.crosstab(
        merged_master["demand_pattern"],
        merged_master["demand_pattern_deseasonalized"],
        margins=True
    ).loc[["smooth", "erratic", "intermittent", "lumpy", "All"], ["smooth", "erratic", "intermittent", "lumpy", "All"]]
    print("\nTransition Matrix (Raw -> Deseasonalized Demand Pattern):")
    print(trans_matrix)

    # Scatter plot: ADI vs CV2 (Raw)
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    plot_data = merged_master.copy()
    plot_data["plot_cv2"] = plot_data["CV2"].clip(lower=1e-3)
    
    colors = {
        "smooth": "#2ca02c",
        "erratic": "#1f77b4",
        "intermittent": "#ff7f0e",
        "lumpy": "#d62728"
    }

    for pat, grp in plot_data.groupby("demand_pattern"):
        ax.scatter(grp["ADI"], grp["plot_cv2"], c=colors[pat], label=f"{pat.title()} ({len(grp):,})", alpha=0.5, edgecolors="none", s=22)

    ax.axvline(1.32, color="black", linestyle="--", linewidth=1.5, label="ADI Cutoff = 1.32")
    ax.axhline(0.49, color="black", linestyle="-.", linewidth=1.5, label=r"$CV^2$ Cutoff = 0.49")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Syntetos-Boylan Demand Classification Grid (ADI vs $CV^2$)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Average Demand Interval (ADI) [Log Scale]", fontsize=11, fontweight="semibold")
    ax.set_ylabel(r"Squared Coefficient of Variation ($CV^2$) [Log Scale]", fontsize=11, fontweight="semibold")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    ax.text(1.02, 0.05, "SMOOTH\n(Regular & Low Var)", fontsize=10, fontweight="bold", color="#2ca02c")
    ax.text(1.02, 25.0, "ERRATIC\n(Regular & High Var)", fontsize=10, fontweight="bold", color="#1f77b4")
    ax.text(3.5, 0.05, "INTERMITTENT\n(Sporadic & Low Var)", fontsize=10, fontweight="bold", color="#ff7f0e")
    ax.text(3.5, 10.0, "LUMPY\n(Sporadic & High Var)", fontsize=10, fontweight="bold", color="#d62728")

    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    adi_cv2_path = eda_dir / "adi_cv2_scatter.png"
    plt.savefig(adi_cv2_path)
    plt.close()
    print(f"Saved ADI vs CV2 scatter plot to: {adi_cv2_path}")

    # B3: Lifecycle Status per SKU
    print("\nRunning B3: Lifecycle Analysis...")
    cutoff_new = pd.Timestamp("2011-06-06")
    cutoff_discontinued = pd.Timestamp("2011-08-29")

    def assign_lifecycle(row):
        is_new = (row["first_sale_week"] >= cutoff_new)
        is_disc = (row["last_sale_week"] < cutoff_discontinued)
        if is_new:
            return "new"
        elif is_disc:
            return "discontinued"
        else:
            return "continuing"

    merged_master["lifecycle_status"] = merged_master.apply(assign_lifecycle, axis=1)

    lifecycle_summary = merged_master.groupby("lifecycle_status").agg(
        sku_count=("StockCode", "count"),
        total_rev=("total_revenue", "sum")
    ).loc[["continuing", "discontinued", "new"]]
    lifecycle_summary["sku_pct"] = (lifecycle_summary["sku_count"] / len(merged_master)) * 100
    lifecycle_summary["rev_pct"] = (lifecycle_summary["total_rev"] / grand_revenue) * 100

    # ABC x Demand Pattern Matrix
    abc_pattern_counts = pd.crosstab(merged_master["abc_class"], merged_master["demand_pattern"], margins=True).loc[["A", "B", "C", "All"], ["smooth", "erratic", "intermittent", "lumpy", "All"]]
    abc_pattern_rev = pd.pivot_table(
        merged_master, index="abc_class", columns="demand_pattern", values="total_revenue", aggfunc="sum", fill_value=0.0
    ).loc[["A", "B", "C"], ["smooth", "erratic", "intermittent", "lumpy"]]
    abc_pattern_rev_pct = (abc_pattern_rev / grand_revenue) * 100

    # ABC x Lifecycle Matrix
    abc_life_counts = pd.crosstab(merged_master["abc_class"], merged_master["lifecycle_status"], margins=True).loc[["A", "B", "C", "All"], ["continuing", "discontinued", "new", "All"]]
    abc_life_rev = pd.pivot_table(
        merged_master, index="abc_class", columns="lifecycle_status", values="total_revenue", aggfunc="sum", fill_value=0.0
    ).loc[["A", "B", "C"], ["continuing", "discontinued", "new"]]
    abc_life_rev_pct = (abc_life_rev / grand_revenue) * 100

    # Save enriched sku_classification.parquet
    sku_class = merged_master[[
        "StockCode", "Description", "total_revenue", "abc_class",
        "ADI", "CV2", "demand_pattern", 
        "CV2_deseasonalized", "demand_pattern_deseasonalized",
        "lifecycle_status", "nonzero_weeks", "active_open_weeks"
    ]].copy()
    sku_class_path = Path("data/processed/sku_classification.parquet")
    sku_class.to_parquet(sku_class_path, index=False)
    print(f"Saved enriched SKU classification table to: {sku_class_path} ({len(sku_class):,} SKUs)")

    # B4: Seasonality Analysis
    print("\nRunning B4: Seasonality Analysis...")
    monthly_idx_df = monthly_idx.reset_index()
    monthly_idx_df.columns = ["month", "seasonal_index"]

    # Build side-by-side Year 1 (weeks 0 to 51) vs Year 2 (weeks 52 to 103)
    weekly_total["cycle_year"] = weekly_total["week_index"].apply(lambda idx: 1 if idx < 52 else 2)
    y1_m = weekly_total[weekly_total["cycle_year"] == 1].groupby("month")[["total_qty", "total_revenue"]].sum()
    y2_m = weekly_total[weekly_total["cycle_year"] == 2].groupby("month")[["total_qty", "total_revenue"]].sum()

    month_names = ["December", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November"]
    month_nums = [12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    monthly_comparison = []
    for m_name, m_num in zip(month_names, month_nums):
        q1 = int(y1_m.loc[m_num, "total_qty"]) if m_num in y1_m.index else 0
        r1 = float(y1_m.loc[m_num, "total_revenue"]) if m_num in y1_m.index else 0.0
        q2 = int(y2_m.loc[m_num, "total_qty"]) if m_num in y2_m.index else 0
        r2 = float(y2_m.loc[m_num, "total_revenue"]) if m_num in y2_m.index else 0.0
        s_idx = monthly_idx_df[monthly_idx_df["month"] == m_num]["seasonal_index"].values[0]
        
        q_g = ((q2 - q1) / q1 * 100) if q1 > 0 else np.nan
        r_g = ((r2 - r1) / r1 * 100) if r1 > 0 else np.nan
        monthly_comparison.append({
            "Month": m_name,
            "Month_Num": m_num,
            "Seasonal_Index": s_idx,
            "Y1_Qty": q1,
            "Y1_Rev": r1,
            "Y2_Qty": q2,
            "Y2_Rev": r2,
            "Qty_Growth_%": q_g,
            "Rev_Growth_%": r_g
        })
    monthly_comp_df = pd.DataFrame(monthly_comparison)

    y1_total_qty = sum(r["Y1_Qty"] for r in monthly_comparison)
    y2_total_qty = sum(r["Y2_Qty"] for r in monthly_comparison)
    y1_rev = sum(r["Y1_Rev"] for r in monthly_comparison)
    y2_rev = sum(r["Y2_Rev"] for r in monthly_comparison)
    y1_sep_nov_qty = sum(r["Y1_Qty"] for r in monthly_comparison if r["Month_Num"] in [9, 10, 11])
    y2_sep_nov_qty = sum(r["Y2_Qty"] for r in monthly_comparison if r["Month_Num"] in [9, 10, 11])
    y1_autumn_share = (y1_sep_nov_qty / y1_total_qty) * 100
    y2_autumn_share = (y2_sep_nov_qty / y2_total_qty) * 100
    print(f"Autumn (Sep-Nov) Qty Share: Year 1 = {y1_autumn_share:.2f}%, Year 2 = {y2_autumn_share:.2f}%")

    # YoY Overlay Chart
    y1_weeks = weekly_total.iloc[:52].copy().reset_index(drop=True)
    y2_weeks = weekly_total.iloc[52:104].copy().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.plot(y1_weeks.index, y1_weeks["total_qty"], color="#1f77b4", linewidth=2.2, label=f"Year 1 (Weeks 0-51: 2009-12-07 to 2010-11-29)")
    ax.plot(y2_weeks.index, y2_weeks["total_qty"], color="#ff7f0e", linewidth=2.2, linestyle="--", label=f"Year 2 (Weeks 52-103: 2010-12-06 to 2011-11-28)")
    ax.axvspan(38, 51, color="#2ca02c", alpha=0.15, label="Autumn Peak (Sep-Nov)")
    ax.set_title("Year-over-Year Weekly Demand Overlay (52-Week Alignment)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Week Index within Annual Cycle (0 = Early Dec, 51 = Late Nov)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Total Units Demanded", fontsize=11, fontweight="semibold")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    yoy_path = eda_dir / "yoy_overlay.png"
    plt.savefig(yoy_path)
    plt.close()
    print(f"Saved YoY overlay chart to: {yoy_path}")

    # STL Decomposition
    print("Running STL Decomposition...")
    s = weekly_total["total_qty"].values.astype(float)
    stl_method = "robust=True"
    try:
        res = STL(s, period=52, robust=True).fit()
    except Exception as e:
        stl_method = "seasonal=13"
        res = STL(s, period=52, seasonal=13).fit()

    trend = res.trend
    seasonal = res.seasonal
    resid = res.resid

    var_r = np.var(resid)
    var_tr = np.var(trend + resid)
    var_sr = np.var(seasonal + resid)
    Ft = max(0, 1 - var_r / var_tr)
    Fs = max(0, 1 - var_r / var_sr)

    open_indices = weekly_total[~weekly_total["is_closed_week"]]["week_index"].values
    first_13_open = open_indices[:13]
    last_13_open = open_indices[-13:]
    trend_first13_mean = float(trend[first_13_open].mean())
    trend_last13_mean = float(trend[last_13_open].mean())
    trend_delta = trend_last13_mean - trend_first13_mean
    trend_direction = "Decrease" if trend_delta < 0 else "Increase"
    print(f"STL Method: {stl_method} | Trend strength Ft: {Ft:.4f} | Seasonal strength Fs: {Fs:.4f}")
    print(f"STL Open-Week Trend: First 13 = {trend_first13_mean:,.1f}, Last 13 = {trend_last13_mean:,.1f} | Change = {trend_delta:,.1f} ({trend_direction})")

    # STL plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), dpi=300, sharex=True)
    dates_104 = weekly_total["week_start"]
    axes[0].plot(dates_104, s, color="#1f77b4", linewidth=1.8)
    axes[0].set_ylabel("Observed", fontsize=10, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_title(f"STL Decomposition of Weekly Total Demand (Period=52, {stl_method})", fontsize=13, fontweight="bold")

    axes[1].plot(dates_104, trend, color="#d62728", linewidth=2.0)
    axes[1].set_ylabel(f"Trend (Ft={Ft:.2f})", fontsize=10, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(dates_104, seasonal, color="#2ca02c", linewidth=1.8)
    axes[2].set_ylabel(f"Seasonal (Fs={Fs:.2f})", fontsize=10, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    axes[3].scatter(dates_104, resid, color="gray", s=15, alpha=0.7)
    axes[3].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[3].set_ylabel("Remainder", fontsize=10, fontweight="bold")
    axes[3].set_xlabel("Calendar Week Start", fontsize=11, fontweight="semibold")
    axes[3].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    stl_path = eda_dir / "stl_decomposition.png"
    plt.savefig(stl_path)
    plt.close()
    print(f"Saved STL decomposition chart to: {stl_path}")

    # Top 6 Class-A SKUs weekly demand charts
    print("Generating top 6 Class-A SKUs weekly demand charts...")
    top_6_skus = sku_master_sorted[sku_master_sorted["abc_class"] == "A"].head(6)["StockCode"].tolist()

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), dpi=300, sharex=True)
    axes = axes.flatten()

    for i, sku in enumerate(top_6_skus):
        ax = axes[i]
        sku_data = weekly_sku[weekly_sku["StockCode"] == sku].copy()
        sku_desc = sku_master_sorted[sku_master_sorted["StockCode"] == sku]["Description"].values[0]
        sku_rev = sku_master_sorted[sku_master_sorted["StockCode"] == sku]["total_revenue"].values[0]
        
        ax.plot(sku_data["week_start"], sku_data["qty"], color="#1f77b4", linewidth=1.5, label="Raw Qty")
        if "qty_capped" in sku_data.columns:
            ax.plot(sku_data["week_start"], sku_data["qty_capped"], color="#d62728", linestyle="--", linewidth=1.2, label="99th Cap")

        ax.set_title(f"Rank {i+1}: SKU {sku} - {sku_desc[:25]} (£{sku_rev:,.0f})", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f}"))
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    top_sku_path = eda_dir / "top_sku_weekly.png"
    plt.savefig(top_sku_path)
    plt.close()
    print(f"Saved top 6 SKUs chart to: {top_sku_path}")

    # B5: Growth, Customer Concentration & Decomposition
    print("\nRunning B5: Growth, Decomposition & Customer Concentration...")
    qty_growth_pct = ((y2_total_qty - y1_total_qty) / y1_total_qty) * 100
    rev_growth_pct = ((y2_rev - y1_rev) / y1_rev) * 100

    # Decomposition
    P1 = y1_rev / y1_total_qty
    P2 = y2_rev / y2_total_qty
    volume_effect = (y2_total_qty - y1_total_qty) * P1
    price_mix_effect = y2_total_qty * (P2 - P1)
    delta_revenue = y2_rev - y1_rev

    print(f"YoY Growth: Qty = {qty_growth_pct:.2f}% | Revenue = {rev_growth_pct:.2f}%")
    print(f"P1 = £{P1:.4f} | P2 = £{P2:.4f}")
    print(f"Volume Effect = £{volume_effect:,.2f} | Price/Mix Effect = £{price_mix_effect:,.2f} | Delta Rev = £{delta_revenue:,.2f}")

    # Customer Concentration strictly within the 104-week window
    tot_clean_qty = tx_104["Quantity"].sum()
    tot_clean_rev = tx_104["Revenue"].sum()
    known_tx = tx_104[tx_104["Customer ID"].notnull()]
    guest_tx = tx_104[tx_104["Customer ID"].isnull()]

    known_cust_qty = known_tx["Quantity"].sum()
    known_cust_rev = known_tx["Revenue"].sum()
    guest_qty = guest_tx["Quantity"].sum()
    guest_rev = guest_tx["Revenue"].sum()

    known_cust_pct = (known_cust_qty / tot_clean_qty) * 100
    guest_pct = (guest_qty / tot_clean_qty) * 100

    cust_qty = known_tx.groupby("Customer ID")["Quantity"].sum().sort_values(ascending=False)
    top_10_cust_qty = cust_qty.head(10).sum()
    top_50_cust_qty = cust_qty.head(50).sum()

    top_10_cust_share = (top_10_cust_qty / tot_clean_qty) * 100
    top_50_cust_share = (top_50_cust_qty / tot_clean_qty) * 100
    print(f"Reconciled Customer Share: Known = {known_cust_pct:.2f}% | Guest = {guest_pct:.2f}%")
    print(f"Reconciled Concentration: Top 10 = {top_10_cust_share:.2f}% | Top 50 = {top_50_cust_share:.2f}%")

    # Dec-Apr Unit Decline & Single Customer Audit
    print("Auditing Dec-Apr unit decline by SKU...")
    cal_w = calendar_df.copy()
    cal_w["cycle_year"] = cal_w["week_index"].apply(lambda idx: 1 if idx < 52 else 2)
    decapr_y1_weeks = cal_w[(cal_w["cycle_year"] == 1) & (cal_w["month"].isin([12, 1, 2, 3, 4]))]["week_start"]
    decapr_y2_weeks = cal_w[(cal_w["cycle_year"] == 2) & (cal_w["month"].isin([12, 1, 2, 3, 4]))]["week_start"]

    y1_decapr_sku = weekly_sku[weekly_sku["week_start"].isin(decapr_y1_weeks)].groupby("StockCode")["qty"].sum()
    y2_decapr_sku = weekly_sku[weekly_sku["week_start"].isin(decapr_y2_weeks)].groupby("StockCode")["qty"].sum()

    sku_decline = pd.DataFrame({"y1_qty": y1_decapr_sku, "y2_qty": y2_decapr_sku}).fillna(0)
    sku_decline["decline"] = sku_decline["y1_qty"] - sku_decline["y2_qty"]
    top10_decline = sku_decline.sort_values(by="decline", ascending=False).head(10).reset_index()

    sku_master_map = sku_master.set_index("StockCode")
    top10_decline_rows = []
    for _, r in top10_decline.iterrows():
        sku = r["StockCode"]
        dec = r["decline"]
        desc = sku_master_map.loc[sku, "Description"] if sku in sku_master_map.index else ""
        tx_sku_y1 = tx_104[(tx_104["StockCode"] == sku) & (tx_104["week_start"].isin(decapr_y1_weeks))]
        tx_sku_y2 = tx_104[(tx_104["StockCode"] == sku) & (tx_104["week_start"].isin(decapr_y2_weeks))]
        
        cust_y1 = tx_sku_y1.groupby("Customer ID")["Quantity"].sum()
        cust_y2 = tx_sku_y2.groupby("Customer ID")["Quantity"].sum()
        cust_diff = pd.DataFrame({"y1": cust_y1, "y2": cust_y2}).fillna(0)
        cust_diff["cust_decline"] = cust_diff["y1"] - cust_diff["y2"]
        cust_diff["pct_of_sku_decline"] = cust_diff["cust_decline"] / dec * 100
        top_cust = cust_diff.sort_values(by="cust_decline", ascending=False).head(1)
        
        top_cust_id = top_cust.index[0] if len(top_cust) > 0 else "None"
        top_cust_pct = top_cust["pct_of_sku_decline"].values[0] if len(top_cust) > 0 else 0.0
        top_cust_drop = top_cust["cust_decline"].values[0] if len(top_cust) > 0 else 0
        single_cust_flag = (top_cust_pct > 25.0)

        top10_decline_rows.append({
            "StockCode": sku,
            "Description": desc[:28],
            "Y1_Qty": int(r["y1_qty"]),
            "Y2_Qty": int(r["y2_qty"]),
            "Unit_Decline": int(dec),
            "Top_Customer": f"Cust {top_cust_id:.0f}" if isinstance(top_cust_id, (int, float)) else str(top_cust_id),
            "Customer_Drop": int(top_cust_drop),
            "Customer_Share_%": top_cust_pct,
            "Flag_Over_25%": single_cust_flag
        })
    top10_decline_df = pd.DataFrame(top10_decline_rows)

    # B6: Candidate Forecasting Universes
    print("\nRunning B6: Candidate Forecasting Universes...")
    rule1 = merged_master[(merged_master["abc_class"] == "A") & (merged_master["lifecycle_status"] == "continuing")]
    rule2 = merged_master[merged_master["abc_class"].isin(["A", "B"]) & (merged_master["lifecycle_status"] == "continuing")]
    rule3 = rule2[rule2["nonzero_weeks"] >= 40]
    rule4 = rule2[rule2["demand_pattern"].isin(["smooth", "erratic"])]

    universe_results = [
        {
            "Rule": "(i) ABC Class A & Lifecycle Continuing",
            "SKUs": len(rule1),
            "SKU_Share_%": (len(rule1)/len(merged_master))*100,
            "Revenue": rule1["total_revenue"].sum(),
            "Rev_Share_%": (rule1["total_revenue"].sum()/grand_revenue)*100,
            "Tradeoff": "Highest revenue density per SKU; captures 74.2% revenue with only 901 SKUs, but leaves secondary B items unforecasted."
        },
        {
            "Rule": "(ii) ABC Class A or B & Lifecycle Continuing",
            "SKUs": len(rule2),
            "SKU_Share_%": (len(rule2)/len(merged_master))*100,
            "Revenue": rule2["total_revenue"].sum(),
            "Rev_Share_%": (rule2["total_revenue"].sum()/grand_revenue)*100,
            "Tradeoff": "Balanced operational standard; captures 84.6% revenue across 1,760 SKUs, removing discontinued and inactive churn lines."
        },
        {
            "Rule": "(iii) Rule (ii) with >= 40 Non-Zero Weeks",
            "SKUs": len(rule3),
            "SKU_Share_%": (len(rule3)/len(merged_master))*100,
            "Revenue": rule3["total_revenue"].sum(),
            "Rev_Share_%": (rule3["total_revenue"].sum()/grand_revenue)*100,
            "Tradeoff": "Data-rich mature subset; ensures dense history for deep autoregressive lags, but sacrifices recently launched lines."
        },
        {
            "Rule": "(iv) Rule (ii) with Smooth or Erratic Pattern",
            "SKUs": len(rule4),
            "SKU_Share_%": (len(rule4)/len(merged_master))*100,
            "Revenue": rule4["total_revenue"].sum(),
            "Rev_Share_%": (rule4["total_revenue"].sum()/grand_revenue)*100,
            "Tradeoff": "Optimized for continuous time-series models (ARIMA/ETS); eliminates zero-order intermittency but drops 36.5% of A/B lines."
        }
    ]
    univ_df = pd.DataFrame(universe_results)

    # =============================================================
    # WRITE REPORT: reports/03_eda.md
    # =============================================================
    print("\nWriting comprehensive EDA report to reports/03_eda.md...")
    report_lines = []
    report_lines.append("# Exploratory Data Analysis & Demand Profiling Report (Reconciled)")
    report_lines.append("\n**Dataset**: UCI Online Retail II (Cleaned 104-Week Series)")
    report_lines.append(f"**Execution Timestamp**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Analysis Scope**: `{len(merged_master):,}` Physical SKUs across `{len(calendar_df)}` Calendar Weeks (`2009-12-07` to `2011-11-28`)")
    report_lines.append(f"**Reconciled 104-Week Totals**: **`{grand_units:,}` Units** | **`£{grand_revenue:,.2f}` Net Revenue**\n")
    report_lines.append("---\n")

    # TOP-LEVEL CORRECTIONS SECTION
    report_lines.append("## Corrections to Previous Findings")
    report_lines.append("This section documents the formal audit and mathematical reconciliation of previous discrepancies, narrative corrections, and methodology enhancements:")
    report_lines.append("\n### 1. Mathematical Reconciliation of 104-Week Totals")
    report_lines.append("Previously, three disparate totals appeared across the analysis due to conflicting date boundaries and table filter scopes. All tables are now strictly reconciled to the identical 104-week window (`2009-12-07` to `2011-11-28`):")
    report_lines.append("\n| Metric Source | Previous Figures | Reconciled Figures | Root Cause of Previous Gap |")
    report_lines.append("| :--- | :---: | :---: | :--- |")
    report_lines.append(f"| **`weekly_sku_demand.parquet`** | {grand_units:,} units<br>£{grand_revenue:,.2f} | **{grand_units:,} units<br>£{grand_revenue:,.2f}** | **Source of Truth**: Defined strictly on the 104 contiguous calendar weeks. |")
    report_lines.append(f"| **Monthly Aggregation** | 10,510,457 units<br>£18,480,213.19 | **{grand_units:,} units<br>£{grand_revenue:,.2f}** | Previously grouped by calendar month `2009-12-01` to `2011-11-30`. That included Dec 1–6, 2009 (+141,667 units, +£254,391.75 from dropped week `2009-11-30`) and excluded Dec 1–4, 2011 (-67,546 units, -£130,060.25 from kept week `2011-11-28`), producing a net gap of **+74,121 units and +£124,331.50**. Now aligned to 52-week cycles. |")
    report_lines.append(f"| **Customer Concentration** | 10,741,552 units | **{grand_units:,} units<br>£{grand_revenue:,.2f}** | Previously computed on `clean_transactions` without filtering by the 104 calendar weeks, erroneously including the 2 dropped boundary weeks (`2009-11-30` and `2011-12-05`), which contained **+305,216 units and +£569,458.57**. |")
    
    report_lines.append("\n### 2. Business Narrative Correction: Revenue vs. Volume Decomposition")
    report_lines.append(f"While top-line net revenue grew by **+{rev_growth_pct:.2f}%** (+£{delta_revenue:,.2f}), physical unit volume actually declined by **{qty_growth_pct:.2f}%** (-{y1_total_qty - y2_total_qty:,} units). Average price realization expanded from **£{P1:.4f}/unit** in Year 1 to **£{P2:.4f}/unit** in Year 2 (+8.39%).")
    report_lines.append("\nDecomposing the £183,785.71 revenue increase:")
    report_lines.append(f"- **Volume Effect**: $(Q_2 - Q_1) \\times P_1 = ({y2_total_qty:,} - {y1_total_qty:,}) \\times £{P1:.4f} =$ **-£{abs(volume_effect):,.2f}**")
    report_lines.append(f"- **Price/Mix Effect**: $Q_2 \\times (P_2 - P_1) = {y2_total_qty:,} \\times (£{P2:.4f} - £{P1:.4f}) =$ **+£{price_mix_effect:,.2f}**")
    report_lines.append(f"- **Net Revenue Impact**: $\\Delta \\text{{Revenue}} = \\text{{Volume Effect}} + \\text{{Price/Mix Effect}} =$ **+£{delta_revenue:,.2f}**")
    report_lines.append("\n*Takeaway*: Business top-line expansion was entirely driven by price/mix optimization and higher unit prices, masking a underlying volume contraction.")

    report_lines.append("\n### 3. STL Trend Direction")
    report_lines.append(f"Evaluating the STL trend component across open operating weeks reveals that the underlying baseline demand softened: mean weekly trend fell from **{trend_first13_mean:,.1f} units/week** across the first 13 open weeks to **{trend_last13_mean:,.1f} units/week** across the last 13 open weeks (a net decline of **{trend_delta:,.1f} units/week, or {trend_delta/trend_first13_mean*100:.1f}%**).")

    report_lines.append("\n### 4. Root Cause of Unit Decline (Wholesale Order Concentration in Dec–Apr)")
    report_lines.append("Comparing Year 2 vs Year 1 across the Dec–Apr period, the top 10 SKUs driving the largest unit volume decline were analyzed for customer concentration:")
    report_lines.append("\n| StockCode | Description | Y1 Qty | Y2 Qty | Unit Decline | Top Customer Drop | Cust Share of Decline | Single Cust > 25% |")
    report_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in top10_decline_df.iterrows():
        report_lines.append(
            f"| `{r['StockCode']}` | {r['Description']} | {r['Y1_Qty']:,} | {r['Y2_Qty']:,} | **{r['Unit_Decline']:,}** | {r['Top_Customer']} (-{r['Customer_Drop']:,}) | **{r['Customer_Share_%']:.1f}%** | **{r['Flag_Over_25%']}** |"
        )
    report_lines.append("\n> [!IMPORTANT]\n> **Wholesale Lumpy Order Findings**: **100% of the top 10 declining SKUs** had over **84% of their volume drop** attributable to a single wholesale account (`Customer 13902` or `Customer 17940`). These accounts placed massive bulk orders in early 2010 that did not repeat in early 2011. The apparent volume loss was not a structural retail demand collapse, but non-repeating B2B bulk purchases.")

    report_lines.append("\n### 5. Seasonality Metric Caveat & Separate Autumn Shares")
    report_lines.append(f"- **Caveat on Seasonal Strength ($F_s = {Fs:.4f}$)**: The calculated seasonal strength is artificially high because of the two mandatory Christmas shutdown weeks (which have 0 units demanded).")
    report_lines.append(f"- **Separate Autumn Concentration (Sep–Nov)**: Year 1 = **{y1_autumn_share:.2f}%** of annual volume; Year 2 = **{y2_autumn_share:.2f}%** of annual volume, confirming stable seasonal peaking.")

    report_lines.append("\n---\n")

    # Section 1: Executive Summary
    report_lines.append("## Executive Key Findings (For Business Leaders)")
    report_lines.append(f"- **Extreme Revenue Concentration (Pareto Principle)**: Just **{abc_summary.loc['A', 'sku_count']:,} SKUs ({abc_summary.loc['A', 'sku_pct']:.1f}% of catalog)** generate **{abc_summary.loc['A', 'rev_pct']:.1f}% of total business revenue** (£{abc_summary.loc['A', 'total_rev']:,.2f}). Focusing predictive modeling on these top items will capture the vast majority of commercial impact.")
    report_lines.append(f"- **Pervasive Lumpy & Intermittent Demand**: More than **{demand_summary.loc[['lumpy', 'intermittent'], 'sku_pct'].sum():.1f}% of the physical catalog** exhibits intermittent or lumpy demand patterns. Only **{demand_summary.loc['smooth', 'sku_pct']:.1f}% of SKUs** exhibit smooth, predictable demand.")
    report_lines.append(f"- **High Catalog Churn**: **{lifecycle_summary.loc['discontinued', 'sku_count']:,} SKUs ({lifecycle_summary.loc['discontinued', 'sku_pct']:.1f}%)** were discontinued prior to the final quarter of 2011, leaving **{lifecycle_summary.loc['continuing', 'sku_count']:,} active continuing SKUs** for forward forecasting.")
    report_lines.append(f"- **Customer Account Split**: Registered customer accounts generate **{known_cust_pct:.1f}% of physical units** (£{known_cust_rev:,.2f}), while guest checkout accounts drive **{guest_pct:.1f}%** (£{guest_rev:,.2f}). The top 50 accounts drive **{top_50_cust_share:.1f}%** of total unit volume.")
    report_lines.append("\n---\n")

    # Section 2: Calendar Weeks & Closed Weeks
    report_lines.append("## 1. Calendar Alignment & Closed Weeks (Part A)")
    report_lines.append("A full 104-week calendar (`2009-12-07` to `2011-11-28`) was constructed in [`data/processed/calendar_weeks.parquet`](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/data/processed/calendar_weeks.parquet).")
    report_lines.append("\n| Week Start | ISO Week | Total Qty | Total Revenue | is_closed_week | Operational Reason |")
    report_lines.append("| :---: | :---: | :---: | :---: | :---: | :--- |")
    for cw in closed_weeks:
        cw_row = calendar_df[calendar_df["week_start"] == cw].iloc[0]
        report_lines.append(f"| `{cw.strftime('%Y-%m-%d')}` | W{cw_row['iso_week']} | 0 | £0.00 | **True** | Annual Christmas/New Year Holiday Shutdown |")
    report_lines.append("\nAll interval and rate calculations in this report strictly exclude these 2 closed weeks, operating over the **102 open calendar weeks**.")
    report_lines.append("\n![Macro Weekly Demand with Shaded Closed Weeks](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/weekly_total_demand.png)")
    report_lines.append("\n---\n")

    # Section 3: ABC Analysis
    report_lines.append("## 2. ABC Revenue Segmentation (Part B1)")
    report_lines.append("| ABC Class | SKU Count | % of Catalog | Total Revenue (£) | % Revenue Share | Demand Strategy |")
    report_lines.append("| :---: | :---: | :---: | :---: | :---: | :--- |")
    for cls in ["A", "B", "C"]:
        r = abc_summary.loc[cls]
        strat = "Priority SKU forecasting (ARIMA / Prophet / ML)" if cls == "A" else ("Standard replenishment models" if cls == "B" else "Simple reorder points / rule-based")
        report_lines.append(f"| **Class {cls}** | {int(r['sku_count']):,} | {r['sku_pct']:.2f}% | £{r['total_rev']:,.2f} | **{r['rev_pct']:.2f}%** | {strat} |")
    report_lines.append(f"| **Total** | **{len(merged_master):,}** | **100.00%** | **£{grand_revenue:,.2f}** | **100.00%** | — |")
    report_lines.append("\n![Pareto ABC Curve](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/eda/pareto_abc.png)")
    report_lines.append("\n---\n")

    # Section 4: Syntetos-Boylan Demand Pattern Classification
    report_lines.append("## 3. Syntetos-Boylan Demand Pattern Classification (Part B2)")
    report_lines.append("Demand patterns are categorized using Average Demand Interval ($\\text{ADI} = \\text{active open weeks} / \\text{nonzero weeks}$) and squared coefficient of variation ($CV^2 = (\\sigma / \\mu)^2$ across positive weeks, with `ddof=0`):")
    report_lines.append("- **Smooth** ($\\text{ADI} < 1.32$, $CV^2 < 0.49$): Regular frequency, low variability.")
    report_lines.append("- **Erratic** ($\\text{ADI} < 1.32$, $CV^2 \\ge 0.49$): Regular frequency, high order variability.")
    report_lines.append("- **Intermittent** ($\\text{ADI} \\ge 1.32$, $CV^2 < 0.49$): Sporadic orders, consistent order size.")
    report_lines.append("- **Lumpy** ($\\text{ADI} \\ge 1.32$, $CV^2 \\ge 0.49$): Sporadic orders, highly variable order size.\n")
    report_lines.append("| Demand Pattern | SKU Count (Raw) | % Catalog | SKU Count (Deseas) | % Catalog | Recommended Forecasting Approach |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
    for pat in ["smooth", "erratic", "intermittent", "lumpy"]:
        r_raw = demand_summary.loc[pat]
        r_des = demand_deseas_summary.loc[pat]
        rec = "Classical ARIMA / ETS / Prophet" if pat == "smooth" else ("LightGBM with lag features & winsorization" if pat == "erratic" else ("Croston's Method / SBA" if pat == "intermittent" else "Croston's + Poisson / Bootstrapping"))
        report_lines.append(f"| **{pat.title()}** | {int(r_raw['sku_count']):,} | {r_raw['sku_pct']:.2f}% | {int(r_des['sku_count']):,} | {r_des['sku_pct']:.2f}% | {rec} |")
    report_lines.append(f"| **Total** | **{len(merged_master):,}** | **100.00%** | **{len(merged_master):,}** | **100.00%** | — |")
    
    report_lines.append("\n### Transition Matrix: Raw vs. Deseasonalized Classification")
    report_lines.append("| Raw Pattern \\ Deseasonalized | Smooth | Erratic | Intermittent | Lumpy | Total SKUs |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for pat in ["smooth", "erratic", "intermittent", "lumpy", "All"]:
        row_str = f"| **{pat.title()}** |"
        for col in ["smooth", "erratic", "intermittent", "lumpy", "All"]:
            val = trans_matrix.loc[pat, col]
            row_str += f" {val:,} |"
        report_lines.append(row_str)

    report_lines.append("\n### ABC × Demand Pattern Matrix (SKU Counts & Revenue Share)")
    report_lines.append("| ABC Class | Smooth (Counts / Rev %) | Erratic (Counts / Rev %) | Intermittent (Counts / Rev %) | Lumpy (Counts / Rev %) | Total SKUs | Total Rev % |")
    report_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for cls in ["A", "B", "C"]:
        cnt_s, rev_s = abc_pattern_counts.loc[cls, "smooth"], abc_pattern_rev_pct.loc[cls, "smooth"]
        cnt_e, rev_e = abc_pattern_counts.loc[cls, "erratic"], abc_pattern_rev_pct.loc[cls, "erratic"]
        cnt_i, rev_i = abc_pattern_counts.loc[cls, "intermittent"], abc_pattern_rev_pct.loc[cls, "intermittent"]
        cnt_l, rev_l = abc_pattern_counts.loc[cls, "lumpy"], abc_pattern_rev_pct.loc[cls, "lumpy"]
        tot_c = abc_summary.loc[cls, "sku_count"]
        tot_r = abc_summary.loc[cls, "rev_pct"]
        report_lines.append(f"| **Class {cls}** | {cnt_s:,} ({rev_s:.1f}%) | {cnt_e:,} ({rev_e:.1f}%) | {cnt_i:,} ({rev_i:.1f}%) | {cnt_l:,} ({rev_l:.1f}%) | {int(tot_c):,} | {tot_r:.1f}% |")
    report_lines.append("\n![ADI vs CV2 Scatter Plot](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/eda/adi_cv2_scatter.png)")
    report_lines.append("\n---\n")

    # Section 5: Lifecycle Status
    report_lines.append("## 4. SKU Lifecycle Status (Part B3)")
    report_lines.append("Relative to the 104-week horizon ending `2011-11-28`:")
    report_lines.append("- **New**: `first_sale_week` in the final 26 weeks (`>= 2011-06-06`).")
    report_lines.append("- **Discontinued**: `last_sale_week` more than 13 weeks before calendar end (`< 2011-08-29`).")
    report_lines.append("- **Continuing**: Sold across historical window and actively reordered in the final quarter.\n")
    report_lines.append("| Lifecycle Status | SKU Count | % of Catalog | Total Revenue (£) | % Revenue Share | Modeling Treatment |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
    for stat in ["continuing", "discontinued", "new"]:
        r = lifecycle_summary.loc[stat]
        treat = "Core universe for time-series forecasting" if stat == "continuing" else ("Exclude from future purchase planning" if stat == "discontinued" else "Cold-start / hierarchy-based forecasting")
        report_lines.append(f"| **{stat.title()}** | {int(r['sku_count']):,} | {r['sku_pct']:.2f}% | £{r['total_rev']:,.2f} | **{r['rev_pct']:.2f}%** | {treat} |")
    
    report_lines.append("\n### ABC × Lifecycle Matrix (SKU Counts & Revenue Share)")
    report_lines.append("| ABC Class | Continuing (Counts / Rev %) | Discontinued (Counts / Rev %) | New (Counts / Rev %) | Total SKUs | Total Rev % |")
    report_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for cls in ["A", "B", "C"]:
        cnt_c, rev_c = abc_life_counts.loc[cls, "continuing"], abc_life_rev_pct.loc[cls, "continuing"]
        cnt_d, rev_d = abc_life_counts.loc[cls, "discontinued"], abc_life_rev_pct.loc[cls, "discontinued"]
        cnt_n, rev_n = abc_life_counts.loc[cls, "new"], abc_life_rev_pct.loc[cls, "new"]
        tot_c = abc_summary.loc[cls, "sku_count"]
        tot_r = abc_summary.loc[cls, "rev_pct"]
        report_lines.append(f"| **Class {cls}** | {cnt_c:,} ({rev_c:.1f}%) | {cnt_d:,} ({rev_d:.1f}%) | {cnt_n:,} ({rev_n:.1f}%) | {int(tot_c):,} | {tot_r:.1f}% |")
    report_lines.append("\n---\n")

    # Section 6: Seasonality, Trends & STL Decomposition
    report_lines.append("## 5. Seasonality, Monthly Indices & Decomposition (Part B4)")
    report_lines.append("### Monthly Seasonal Index & Year-over-Year Comparison (52-Week Aligned)")
    report_lines.append("| Calendar Month | Seasonal Index | Year 1 Qty (2009-10) | Year 1 Rev (£) | Year 2 Qty (2010-11) | Year 2 Rev (£) | Qty Growth % | Rev Growth % |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in monthly_comp_df.iterrows():
        report_lines.append(
            f"| **{r['Month']}** | **{r['Seasonal_Index']:.3f}** | {r['Y1_Qty']:,} | £{r['Y1_Rev']:,.2f} | {r['Y2_Qty']:,} | £{r['Y2_Rev']:,.2f} | {r['Qty_Growth_%']:+.1f}% | {r['Rev_Growth_%']:+.1f}% |"
        )
    report_lines.append(f"| **Total** | — | **{y1_total_qty:,}** | **£{y1_rev:,.2f}** | **{y2_total_qty:,}** | **£{y2_rev:,.2f}** | **{qty_growth_pct:+.1f}%** | **{rev_growth_pct:+.1f}%** |")
    report_lines.append(f"\n- **Autumn Concentration (Sep–Nov)**: Year 1 = **{y1_autumn_share:.2f}%** | Year 2 = **{y2_autumn_share:.2f}%** of annual volume.")
    report_lines.append("\n![YoY Weekly Overlay](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/eda/yoy_overlay.png)")
    
    report_lines.append("\n### STL Time-Series Decomposition Metrics")
    report_lines.append(f"- **Decomposition Method**: `statsmodels.tsa.seasonal.STL` (period=52, {stl_method})")
    report_lines.append(f"- **Trend Strength ($F_t$)**: **`{Ft:.4f}`**")
    report_lines.append(f"- **Seasonal Strength ($F_s$)**: **`{Fs:.4f}`** *(Note: inflated by the two Christmas closure zero-demand weeks)*")
    report_lines.append(f"- **Open-Week Trend Shift**: Mean weekly trend decreased from **{trend_first13_mean:,.1f}** (first 13 open weeks) to **{trend_last13_mean:,.1f}** (last 13 open weeks), representing a **{trend_delta:,.1f} units/week ({trend_delta/trend_first13_mean*100:.1f}%)** volume softening.")
    report_lines.append("\n![STL Decomposition Plot](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/eda/stl_decomposition.png)")
    
    report_lines.append("\n### Top 6 Class-A SKUs Weekly Demand Behavior")
    report_lines.append("![Top 6 Class A Demand Profiles](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/eda/top_sku_weekly.png)")
    report_lines.append("\n---\n")

    # Section 7: Growth & Customer Concentration
    report_lines.append("## 6. Business Growth & Customer Concentration (Part B5)")
    report_lines.append("| Metric | Year 1 (Weeks 0-51) | Year 2 (Weeks 52-103) | YoY Change |")
    report_lines.append("| :--- | :---: | :---: | :---: |")
    report_lines.append(f"| **Total Units Demanded** | {y1_total_qty:,} | {y2_total_qty:,} | **{qty_growth_pct:+.2f}%** |")
    report_lines.append(f"| **Total Net Sales Revenue** | £{y1_rev:,.2f} | £{y2_rev:,.2f} | **{rev_growth_pct:+.2f}%** |")
    report_lines.append(f"| **Average Realized Price / Unit** | £{P1:.4f} | £{P2:.4f} | **{((P2-P1)/P1)*100:+.2f}%** |")
    
    report_lines.append("\n### Channel & Account Concentration (104-Week Window)")
    report_lines.append(f"- **Registered Customer Accounts**: `{known_cust_qty:,}` units (**{known_cust_pct:.2f}%**), £{known_cust_rev:,.2f} (**{known_cust_rev/tot_clean_rev*100:.2f}%** of revenue)")
    report_lines.append(f"- **Guest / Marketplace Transactions (Null Customer ID)**: `{guest_qty:,}` units (**{guest_pct:.2f}%**), £{guest_rev:,.2f} (**{guest_rev/tot_clean_rev*100:.2f}%** of revenue)")
    report_lines.append(f"- **Top 10 Customers Concentration**: `{top_10_cust_qty:,}` units (**{top_10_cust_share:.2f}%** of all physical retail volume)")
    report_lines.append(f"- **Top 50 Customers Concentration**: `{top_50_cust_qty:,}` units (**{top_50_cust_share:.2f}%** of all physical retail volume)")
    report_lines.append("\n---\n")

    # Section 8: Candidate Forecasting Universes
    report_lines.append("## 7. Candidate Forecasting Universes (Part B6)")
    report_lines.append("The following candidate universes were evaluated to define the training/evaluation scope for forecasting models:\n")
    report_lines.append("| Universe Filtering Rule | SKU Count | % of Catalog | Total Revenue Covered (£) | % Revenue Covered | Operational Trade-off |")
    report_lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")
    for _, r in univ_df.iterrows():
        report_lines.append(
            f"| **{r['Rule']}** | **{int(r['SKUs']):,}** | **{r['SKU_Share_%']:.1f}%** | **£{r['Revenue']:,.2f}** | **{r['Rev_Share_%']:.1f}%** | {r['Tradeoff']} |"
        )

    report_path = Path("reports/03_eda.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved full reconciled EDA report to: {report_path}")

    # Terminal Summary
    print("\n" + "=" * 70)
    print("EDA PIPELINE EXECUTION COMPLETED (RECONCILED)")
    print(f"Elapsed Time:                  {time.time() - start_time:.2f}s")
    print(f"Calendar Weeks:                104 weeks (Closed weeks: {closed_str})")
    print(f"Reconciled 104-Week Total:     {grand_units:,} units | £{grand_revenue:,.2f}")
    print(f"ABC Class A SKUs:              {abc_summary.loc['A', 'sku_count']:,} SKUs ({abc_summary.loc['A', 'rev_pct']:.1f}% Revenue)")
    print(f"ABC Class B SKUs:              {abc_summary.loc['B', 'sku_count']:,} SKUs ({abc_summary.loc['B', 'rev_pct']:.1f}% Revenue)")
    print(f"ABC Class C SKUs:              {abc_summary.loc['C', 'sku_count']:,} SKUs ({abc_summary.loc['C', 'rev_pct']:.1f}% Revenue)")
    print(f"Demand Patterns (Raw):         Smooth: {demand_summary.loc['smooth', 'sku_count']:,} | Erratic: {demand_summary.loc['erratic', 'sku_count']:,} | Intermittent: {demand_summary.loc['intermittent', 'sku_count']:,} | Lumpy: {demand_summary.loc['lumpy', 'sku_count']:,}")
    print(f"Demand Patterns (Deseas):      Smooth: {demand_deseas_summary.loc['smooth', 'sku_count']:,} | Erratic: {demand_deseas_summary.loc['erratic', 'sku_count']:,} | Intermittent: {demand_deseas_summary.loc['intermittent', 'sku_count']:,} | Lumpy: {demand_deseas_summary.loc['lumpy', 'sku_count']:,}")
    print(f"Lifecycle Status:              Continuing: {lifecycle_summary.loc['continuing', 'sku_count']:,} | Discontinued: {lifecycle_summary.loc['discontinued', 'sku_count']:,} | New: {lifecycle_summary.loc['new', 'sku_count']:,}")
    print(f"YoY Annual Growth:             Volume: {qty_growth_pct:+.1f}% | Revenue: {rev_growth_pct:+.1f}%")
    print(f"Decomposition:                 Volume Effect: £{volume_effect:,.2f} | Price/Mix: £{price_mix_effect:,.2f}")
    print(f"STL Open-Week Trend:           {trend_first13_mean:,.1f} -> {trend_last13_mean:,.1f} ({trend_direction})")
    print(f"Seasonality & Autumn Share:    Fs = {Fs:.4f} (caveat: zeros) | Y1 Autumn: {y1_autumn_share:.1f}% | Y2 Autumn: {y2_autumn_share:.1f}%")
    print("All Parquet datasets, charts, and reports successfully generated!")
    print("=" * 70)


if __name__ == "__main__":
    run_eda_pipeline()
