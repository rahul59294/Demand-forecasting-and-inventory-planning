"""
Wholesale/Retail Demand Split & Retail Baseline Backtest Pipeline (Part A).
A1: Identify wholesale-influenced transaction weeks based on customer concentration (>40% volume share)
    and order volume (>= 3x median non-zero weekly demand).
A2: Construct retail_qty series for all forecast_universe SKUs, capping wholesale spikes at the 95th
    percentile of non-wholesale weeks. Save data/processed/weekly_sku_demand_retail.parquet.
A3: Re-run all 9 baseline models from Milestone 4 across the 3 rolling folds using retail_qty.
    Evaluate against raw retail_qty actuals and save data/processed/backtest_metrics_retail.parquet.
"""

import os
from pathlib import Path
import time
import subprocess

# Matplotlib configuration for sandbox
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

# Model implementations from baseline engine
def model_naive(train_series, horizon=13):
    last_val = train_series[-1] if len(train_series) > 0 else 0.0
    return np.full(horizon, max(0.0, float(last_val)))

def model_ma(train_series, window=4, horizon=13):
    if len(train_series) == 0:
        return np.zeros(horizon)
    sub = train_series[-window:] if len(train_series) >= window else train_series
    val = float(np.mean(sub))
    return np.full(horizon, max(0.0, val))

def fit_ses_level(train_series, alpha_grid=np.linspace(0.02, 0.98, 49)):
    n = len(train_series)
    if n == 0:
        return 0.0
    if n == 1:
        return float(train_series[0])
    best_sse = np.inf
    best_level = float(train_series[-1])
    y = train_series.astype(float)
    for alpha in alpha_grid:
        level = y[0]
        sse = 0.0
        for i in range(1, n):
            level = alpha * y[i] + (1.0 - alpha) * level
            sse += (y[i] - level) ** 2
        if sse < best_sse:
            best_sse = sse
            best_level = level
    return max(0.0, float(best_level))

def model_ses(train_series, horizon=13):
    level = fit_ses_level(train_series)
    return np.full(horizon, level)

def fit_croston_sba(train_series, alpha=0.1):
    n = len(train_series)
    if n == 0:
        return 0.0
    pos_idx = np.where(train_series > 0)[0]
    if len(pos_idx) == 0:
        return 0.0
    z = float(train_series[pos_idx[0]])
    p = 1.0
    q = 0.0
    for val in train_series:
        q += 1.0
        if val > 0:
            z = alpha * float(val) + (1.0 - alpha) * z
            p = alpha * q + (1.0 - alpha) * p
            q = 0.0
    sba_factor = 1.0 - (alpha / 2.0)
    fcst = (z / p) * sba_factor if p > 0 else z
    return max(0.0, float(fcst))

def model_croston(train_series, horizon=13):
    val = fit_croston_sba(train_series, alpha=0.1)
    return np.full(horizon, val)

def fit_tsb(train_series, alpha=0.1, beta=0.1):
    n = len(train_series)
    if n == 0:
        return 0.0
    pos_idx = np.where(train_series > 0)[0]
    if len(pos_idx) == 0:
        return 0.0
    z = float(train_series[pos_idx[0]])
    p = 1.0
    for val in train_series:
        if val > 0:
            z = alpha * float(val) + (1.0 - alpha) * z
            p = beta * 1.0 + (1.0 - beta) * p
        else:
            p = (1.0 - beta) * p
    fcst = z * p
    return max(0.0, float(fcst))

def model_tsb(train_series, horizon=13):
    val = fit_tsb(train_series, alpha=0.1, beta=0.1)
    return np.full(horizon, val)


def run_wholesale_split_pipeline():
    start_time = time.time()
    print("=" * 75)
    print("STARTING WHOLESALE/RETAIL SPLIT & BASELINE RETEST PIPELINE (PART A)")
    print("=" * 75)

    # 1. Load Source Data
    clean_tx = pd.read_parquet("data/processed/clean_transactions.parquet")
    cal = pd.read_parquet("data/processed/calendar_weeks.parquet")
    fu = pd.read_parquet("data/processed/forecast_universe.parquet")
    w_sku = pd.read_parquet("data/processed/weekly_sku_demand.parquet")
    
    fu_skus = set(fu["StockCode"].unique())
    fu_meta = fu.set_index("StockCode").to_dict(orient="index")

    # Restrict clean_transactions to 104-week calendar
    clean_tx["week_start"] = clean_tx["InvoiceDate"].dt.floor("D") - pd.to_timedelta(clean_tx["InvoiceDate"].dt.dayofweek, unit="D")
    tx_104 = clean_tx[clean_tx["week_start"].isin(cal["week_start"])].copy()

    # -------------------------------------------------------------
    # A1: Wholesale-Influenced Weeks Identification
    # -------------------------------------------------------------
    print("\n--- A1: WHOLESALE IDENTIFICATION ---")
    sku_total_qty = tx_104.groupby("StockCode")["Quantity"].sum().to_dict()

    # Customer SKU Volume Shares (Registered Customers)
    tx_reg = tx_104[tx_104["Customer ID"].notnull()].copy()
    cust_sku_qty = tx_reg.groupby(["StockCode", "Customer ID"])["Quantity"].sum().reset_index()
    cust_sku_qty["sku_total"] = cust_sku_qty["StockCode"].map(sku_total_qty)
    cust_sku_qty["cust_share"] = cust_sku_qty["Quantity"] / cust_sku_qty["sku_total"]

    # Customers with > 40% volume share
    ws_customers = cust_sku_qty[cust_sku_qty["cust_share"] > 0.40]
    print(f"Total (StockCode, Customer ID) wholesale pairs (> 40% SKU share): {len(ws_customers):,}")

    # SKU Median Non-Zero Weekly Demand (across open weeks)
    open_sku = w_sku[~w_sku["is_closed_week"]]
    sku_median_pos = open_sku[open_sku["qty"] > 0].groupby("StockCode")["qty"].median().to_dict()

    # Join back to transactions to flag wholesale-influenced weeks
    tx_ws_check = tx_reg.merge(
        ws_customers[["StockCode", "Customer ID", "cust_share"]],
        on=["StockCode", "Customer ID"],
        how="inner"
    )
    tx_ws_check["sku_median_pos"] = tx_ws_check["StockCode"].map(sku_median_pos)
    tx_ws_check["threshold_3x"] = 3.0 * tx_ws_check["sku_median_pos"]
    tx_ws_check["is_ws_order"] = tx_ws_check["Quantity"] >= tx_ws_check["threshold_3x"]

    ws_tx_rows = tx_ws_check[tx_ws_check["is_ws_order"]]
    print(f"Total wholesale-influenced transaction lines: {len(ws_tx_rows):,}")

    # Set of wholesale-influenced (StockCode, week_start)
    ws_weeks_set = set(zip(ws_tx_rows["StockCode"], ws_tx_rows["week_start"]))
    print(f"Total wholesale-influenced SKU-weeks across all SKUs: {len(ws_weeks_set):,}")

    # Filter to Forecast Universe (1,760 SKUs)
    fu_ws_skus = set(sku for sku, w in ws_weeks_set if sku in fu_skus)
    fu_ws_count = len(fu_ws_skus)
    fu_total_skus = len(fu)
    fu_ws_rev = fu[fu["StockCode"].isin(fu_ws_skus)]["total_revenue"].sum()
    fu_total_rev = fu["total_revenue"].sum()
    fu_ws_rev_pct = (fu_ws_rev / fu_total_rev) * 100.0

    print(f"\nForecast Universe Wholesale Impact:")
    print(f"  - Forecast Universe SKUs with >= 1 wholesale-influenced week: {fu_ws_count:,} / {fu_total_skus:,} ({fu_ws_count/fu_total_skus*100:.2f}%)")
    print(f"  - Revenue Covered by Wholesale-Impacted SKUs: £{fu_ws_rev:,.2f} / £{fu_total_rev:,.2f} ({fu_ws_rev_pct:.2f}%)")

    # -------------------------------------------------------------
    # A2: Build weekly_sku_demand_retail.parquet
    # -------------------------------------------------------------
    print("\n--- A2: BUILDING RETAIL_QTY WEEKLY SERIES ---")
    fu_w_sku = w_sku[w_sku["StockCode"].isin(fu_skus)].copy()
    fu_w_sku["wholesale_influenced"] = [
        (sku, w) in ws_weeks_set for sku, w in zip(fu_w_sku["StockCode"], fu_w_sku["week_start"])
    ]

    # Compute retail_qty per SKU
    retail_records = []
    zero_non_ws_skus = []

    for sku, grp in fu_w_sku.groupby("StockCode"):
        grp_sorted = grp.sort_values("week_start").copy()
        non_ws = grp_sorted[~grp_sorted["wholesale_influenced"]]
        
        if len(non_ws) == 0:
            # No non-wholesale weeks available; leave as-is
            zero_non_ws_skus.append(sku)
            grp_sorted["retail_qty"] = grp_sorted["qty"]
        else:
            # Capped at 95th percentile of non-wholesale weeks
            p95_val = float(np.percentile(non_ws["qty"], 95))
            grp_sorted["retail_qty"] = [
                min(float(q), p95_val) if ws_flag else float(q)
                for q, ws_flag in zip(grp_sorted["qty"], grp_sorted["wholesale_influenced"])
            ]
        retail_records.append(grp_sorted)

    retail_df = pd.concat(retail_records, ignore_index=True)
    print(f"SKUs with zero non-wholesale weeks: {len(zero_non_ws_skus)}")

    retail_path = Path("data/processed/weekly_sku_demand_retail.parquet")
    retail_df.to_parquet(retail_path, index=False)
    print(f"Saved weekly retail demand series to: {retail_path} ({len(retail_df):,} rows)")
    
    # Volume reduction summary
    total_raw_qty = retail_df["qty"].sum()
    total_retail_qty = retail_df["retail_qty"].sum()
    vol_diff = total_raw_qty - total_retail_qty
    print(f"Total raw volume: {total_raw_qty:,.0f} units | Total retail volume: {total_retail_qty:,.0f} units")
    print(f"Wholesale volume removed: {vol_diff:,.0f} units ({vol_diff/total_raw_qty*100:.2f}% of universe volume)")

    # -------------------------------------------------------------
    # A3: Re-Run 9 Baseline Models on retail_qty
    # -------------------------------------------------------------
    print("\n--- A3: RE-RUNNING BASELINE BACKTEST ON RETAIL_QTY ---")
    folds_config = [
        {
            "fold": 1,
            "train_end": pd.Timestamp("2011-02-28"),
            "test_start": pd.Timestamp("2011-03-07"),
            "test_end": pd.Timestamp("2011-05-30")
        },
        {
            "fold": 2,
            "train_end": pd.Timestamp("2011-05-30"),
            "test_start": pd.Timestamp("2011-06-06"),
            "test_end": pd.Timestamp("2011-08-29")
        },
        {
            "fold": 3,
            "train_end": pd.Timestamp("2011-08-29"),
            "test_start": pd.Timestamp("2011-09-05"),
            "test_end": pd.Timestamp("2011-11-28")
        }
    ]

    retail_forecast_records = []
    models_list = ["Naive", "MA4", "MA13", "SeasonalNaive52", "SES", "Croston-SBA", "TSB", "SES_seasonal", "Croston_seasonal"]

    for f_cfg in folds_config:
        fold_id = f_cfg["fold"]
        t_end = f_cfg["train_end"]
        t_start = f_cfg["test_start"]
        t_test_end = f_cfg["test_end"]

        test_calendar = cal[(cal["week_start"] >= t_start) & (cal["week_start"] <= t_test_end)].sort_values("week_start").reset_index(drop=True)
        test_weeks_list = test_calendar["week_start"].tolist()

        train_sku_fold = retail_df[(retail_df["week_start"] <= t_end) & (~retail_df["is_closed_week"])].copy()

        # Monthly seasonal factors computed from retail_qty training open weeks
        macro_train_weekly = train_sku_fold.groupby("week_start")["retail_qty"].sum().reset_index()
        macro_train_weekly = macro_train_weekly.merge(cal[["week_start", "month"]], on="week_start", how="left")
        overall_mean_macro = macro_train_weekly["retail_qty"].mean()
        monthly_seasonal_idx = (macro_train_weekly.groupby("month")["retail_qty"].mean() / overall_mean_macro).to_dict()
        test_seasonal_factors = np.array([monthly_seasonal_idx.get(w.month, 1.0) for w in test_weeks_list])

        # Test actuals (retail_qty)
        test_sku_data = retail_df[(retail_df["week_start"] >= t_start) & (retail_df["week_start"] <= t_test_end)]
        sku_test_actuals = {}
        for (sku, w), g in test_sku_data.groupby(["StockCode", "week_start"]):
            sku_test_actuals[(sku, w)] = g["retail_qty"].values[0]

        # Eligibility check on retail_qty: first_sale_week <= test_start - 26w and >= 8 nonzero open training weeks
        cutoff_26w = t_start - pd.Timedelta(weeks=26)
        train_sku_grouped = train_sku_fold.groupby("StockCode")
        train_sku_series_map = {}
        for sku, grp in train_sku_grouped:
            grp_sorted = grp.sort_values("week_start")
            train_sku_series_map[sku] = (grp_sorted["week_start"].values, grp_sorted["retail_qty"].values)

        eligible_skus = []
        for sku in fu["StockCode"]:
            meta = fu_meta[sku]
            if meta["first_sale_week"] > cutoff_26w:
                continue
            if sku not in train_sku_series_map:
                continue
            w_dates, y_vals = train_sku_series_map[sku]
            if np.count_nonzero(y_vals > 0) < 8:
                continue
            eligible_skus.append(sku)

        print(f"Fold {fold_id}: Eligible SKUs on retail_qty = {len(eligible_skus):,} / {len(fu):,}")

        # Pre-index full retail_qty series for SeasonalNaive52 lookup
        sku_retail_full_map = retail_df.groupby("StockCode").apply(
            lambda g: dict(zip(g["week_start"], g["retail_qty"]))
        ).to_dict()

        for sku in eligible_skus:
            w_dates, y_vals_raw = train_sku_series_map[sku]

            # 99th percentile capping on positive training weeks if >= 20 nonzero weeks
            pos_mask = y_vals_raw > 0
            if np.count_nonzero(pos_mask) >= 20:
                cap_val = np.percentile(y_vals_raw[pos_mask], 99)
                y_train = np.minimum(y_vals_raw, cap_val)
            else:
                y_train = y_vals_raw.copy()

            # Deseasonalized training series
            train_months = pd.to_datetime(w_dates).month
            s_factors_train = np.array([monthly_seasonal_idx.get(m, 1.0) for m in train_months])
            y_train_deseas = y_train / s_factors_train

            # In-sample scale factor for MASE
            if len(y_train) > 1:
                scale_denom = np.mean(np.abs(np.diff(y_train)))
                if scale_denom == 0.0:
                    scale_denom = 1.0
            else:
                scale_denom = 1.0

            actual_13 = np.array([sku_test_actuals.get((sku, w), 0.0) for w in test_weeks_list])

            # Models
            fc_naive = model_naive(y_train, 13)
            fc_ma4 = model_ma(y_train, window=4, horizon=13)
            fc_ma13 = model_ma(y_train, window=13, horizon=13)

            fc_sn52 = np.zeros(13)
            w_sku_dict = sku_retail_full_map.get(sku, {})
            for h_idx, tw in enumerate(test_weeks_list):
                target_w52 = tw - pd.Timedelta(weeks=52)
                if target_w52 in w_sku_dict:
                    fc_sn52[h_idx] = float(w_sku_dict[target_w52])
                else:
                    fc_sn52[h_idx] = fc_ma13[h_idx]

            fc_ses = model_ses(y_train, 13)
            fc_croston = model_croston(y_train, 13)
            fc_tsb = model_tsb(y_train, 13)

            ses_deseas_level = fit_ses_level(y_train_deseas)
            fc_ses_seasonal = ses_deseas_level * test_seasonal_factors

            croston_deseas_val = fit_croston_sba(y_train_deseas, alpha=0.1)
            fc_croston_seasonal = croston_deseas_val * test_seasonal_factors

            models_dict = {
                "Naive": fc_naive,
                "MA4": fc_ma4,
                "MA13": fc_ma13,
                "SeasonalNaive52": fc_sn52,
                "SES": fc_ses,
                "Croston-SBA": fc_croston,
                "TSB": fc_tsb,
                "SES_seasonal": fc_ses_seasonal,
                "Croston_seasonal": fc_croston_seasonal
            }

            for m_name, fc_arr in models_dict.items():
                for h_step in range(1, 14):
                    retail_forecast_records.append({
                        "fold": fold_id,
                        "model": m_name,
                        "StockCode": sku,
                        "week_start": test_weeks_list[h_step - 1],
                        "horizon_step": h_step,
                        "forecast": float(fc_arr[h_step - 1]),
                        "actual": float(actual_13[h_step - 1]),
                        "mase_scale": scale_denom
                    })

    retail_fc_df = pd.DataFrame(retail_forecast_records)
    
    # Calculate Metrics
    retail_fc_df = retail_fc_df.merge(
        fu[["StockCode", "abc_class", "demand_pattern", "demand_pattern_deseasonalized", "median_price"]],
        on="StockCode",
        how="left"
    )

    def calc_metrics_df(df):
        sum_act = df["actual"].sum()
        sum_fcst = df["forecast"].sum()
        abs_err = np.abs(df["actual"] - df["forecast"]).sum()
        signed_err = (df["forecast"] - df["actual"]).sum()

        wape = (abs_err / sum_act * 100.0) if sum_act > 0 else 0.0
        bias = (signed_err / sum_act * 100.0) if sum_act > 0 else 0.0

        rev_act = (df["actual"] * df["median_price"]).sum()
        rev_err = (np.abs(df["actual"] - df["forecast"]) * df["median_price"]).sum()
        rev_wape = (rev_err / rev_act * 100.0) if rev_act > 0 else 0.0

        sku_mase = df.groupby("StockCode").apply(
            lambda g: np.mean(np.abs(g["actual"] - g["forecast"])) / g["mase_scale"].iloc[0]
        )
        mean_mase = float(sku_mase.mean())

        return {
            "sum_actual": float(sum_act),
            "sum_forecast": float(sum_fcst),
            "wape": float(wape),
            "bias": float(bias),
            "rev_wape": float(rev_wape),
            "mase": float(mean_mase)
        }

    # Overall comparison on retail_qty
    retail_metrics_list = []
    for m in models_list:
        sub = retail_fc_df[retail_fc_df["model"] == m]
        met = calc_metrics_df(sub)
        met["model"] = m
        retail_metrics_list.append(met)
    retail_metrics_df = pd.DataFrame(retail_metrics_list)[["model", "wape", "bias", "mase", "rev_wape", "sum_actual", "sum_forecast"]]

    # Save to backtest_metrics_retail.parquet
    retail_metrics_path = Path("data/processed/backtest_metrics_retail.parquet")
    retail_metrics_df.to_parquet(retail_metrics_path, index=False)
    print(f"\nSaved retail baseline metrics to: {retail_metrics_path}")

    # Load Milestone 4 original metrics for comparison
    orig_metrics_path = Path("data/processed/backtest_metrics.parquet")
    if orig_metrics_path.exists():
        orig_metrics_df = pd.read_parquet(orig_metrics_path)
        orig_overall = orig_metrics_df[(orig_metrics_df["slice_type"] == "overall") & (orig_metrics_df["slice_val"] == "all")].set_index("model")
        
        comp_df = retail_metrics_df.copy().set_index("model")
        comp_df["orig_wape"] = orig_overall.loc[comp_df.index, "wape"]
        comp_df["orig_bias"] = orig_overall.loc[comp_df.index, "bias"]
        comp_df["orig_mase"] = orig_overall.loc[comp_df.index, "mase"]
        comp_df["wape_diff"] = comp_df["wape"] - comp_df["orig_wape"]
        
        print("\n" + "=" * 80)
        print("SIDE-BY-SIDE COMPARISON: ORIGINAL QTY vs RETAIL_QTY BASELINE LEADERBOARD")
        print("=" * 80)
        print(comp_df[["orig_wape", "wape", "wape_diff", "orig_bias", "bias", "orig_mase", "mase"]].to_string())
        print("=" * 80)

    print(f"Wholesale split & baseline retest execution completed in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    run_wholesale_split_pipeline()
