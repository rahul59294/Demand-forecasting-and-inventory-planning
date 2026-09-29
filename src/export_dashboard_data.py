"""
Export Compact JSON Files for Standalone HTML/JS Dashboard (Milestones 8 & 9)
Generates lightweight JSON files in outputs/dashboard/data/ (and outputs/dashboard_data/).

Files generated:
1. overview.json
2. sku_catalog.json
3. sku_history.json
4. sku_forecast.json
5. inventory_policy.json
6. model_performance.json
7. key_findings.json
8. dashboard_data.zip
"""

import json
import time
import zipfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd


def export_dashboard_data():
    start_time = time.time()
    print("=" * 80)
    print("STARTING EXPORT OF COMPACT JSON DASHBOARD DATA")
    print("=" * 80)

    # Primary target for GitHub Pages / HTML dashboard
    out_dir = Path("outputs/dashboard/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Secondary mirror directory for backward compatibility
    mirror_dir = Path("outputs/dashboard_data")
    mirror_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path("data/processed")

    # 1. Load Data
    print("\n--- 1. LOADING PROCESSED PARQUET DATASETS ---")
    fu = pd.read_parquet(data_dir / "forecast_universe.parquet")
    wt = pd.read_parquet(data_dir / "weekly_total.parquet")
    cal = pd.read_parquet(data_dir / "calendar_weeks.parquet")
    sc = pd.read_parquet(data_dir / "sku_classification.parquet")
    w_sku = pd.read_parquet(data_dir / "weekly_sku_demand_retail.parquet")
    fc_13 = pd.read_parquet(data_dir / "final_forecast_next13weeks.parquet")
    inv = pd.read_parquet(data_dir / "inventory_policy.parquet")

    universe_skus = set(fu["StockCode"].tolist())
    print(f"Loaded Forecast Universe: {len(universe_skus):,} SKUs")
    print(f"Loaded Weekly SKU Demand: {len(w_sku):,} rows")
    print(f"Loaded Final Forecast: {len(fc_13):,} rows")
    print(f"Loaded Inventory Policy: {len(inv):,} rows")

    # -------------------------------------------------------------
    # 2. Build File 1: overview.json
    # -------------------------------------------------------------
    print("\n--- 2. BUILDING overview.json ---")
    closed_weeks = cal[cal["is_closed_week"]]["week_start"].dt.strftime("%Y-%m-%d").tolist()
    catalog_abc = sc["abc_class"].value_counts().to_dict()

    weekly_totals = []
    for _, r in wt.sort_values("week_start").iterrows():
        qty_val = float(r["total_qty"])
        weekly_totals.append({
            "week_start": str(r["week_start"])[:10],
            "qty": int(qty_val) if qty_val.is_integer() else round(qty_val, 1),
            "revenue": round(float(r["total_revenue"]), 2),
            "is_closed_week": bool(r["is_closed_week"])
        })

    tot_qty_val = float(wt["total_qty"].sum())
    overview = {
        "total_revenue": round(float(wt["total_revenue"].sum()), 2),
        "total_units": int(tot_qty_val) if tot_qty_val.is_integer() else round(tot_qty_val, 1),
        "sku_counts": {
            "A": int(catalog_abc.get("A", 0)),
            "B": int(catalog_abc.get("B", 0)),
            "C": int(catalog_abc.get("C", 0))
        },
        "universe_sku_counts": {
            "A": int((fu["abc_class"] == "A").sum()),
            "B": int((fu["abc_class"] == "B").sum())
        },
        "closed_weeks": closed_weeks,
        "weekly_total": weekly_totals
    }

    overview_path = out_dir / "overview.json"
    with open(overview_path, "w", encoding="utf-8") as f:
        json.dump(overview, f, separators=(",", ":"))
    print(f"Saved: {overview_path}")

    # -------------------------------------------------------------
    # 3. Build File 2: sku_catalog.json
    # -------------------------------------------------------------
    print("\n--- 3. BUILDING sku_catalog.json ---")
    catalog = []
    for _, r in fu.sort_values("total_revenue", ascending=False).iterrows():
        catalog.append({
            "stock_code": str(r["StockCode"]),
            "description": str(r["Description"]),
            "abc_class": str(r["abc_class"]),
            "demand_pattern": str(r["demand_pattern"]),
            "lifecycle_status": str(r["lifecycle_status"]),
            "median_price": round(float(r["median_price"]), 2),
            "total_revenue": round(float(r["total_revenue"]), 2),
            "adi": round(float(r["ADI"]), 2),
            "cv2": round(float(r["CV2"]), 2)
        })

    catalog_path = out_dir / "sku_catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, separators=(",", ":"))
    print(f"Saved: {catalog_path} ({len(catalog):,} SKUs)")

    # -------------------------------------------------------------
    # 4. Build File 3: sku_history.json (Last 52 Weeks)
    # -------------------------------------------------------------
    print("\n--- 4. BUILDING sku_history.json (Last 52 Weeks) ---")
    last_52_weeks = cal.sort_values("week_start").tail(52)
    min_w52 = last_52_weeks["week_start"].min()
    print(f"History window: {min_w52.strftime('%Y-%m-%d')} to {last_52_weeks['week_start'].max().strftime('%Y-%m-%d')}")

    # Filter for universe SKUs and last 52 weeks
    w_sku_52 = w_sku[
        (w_sku["StockCode"].isin(universe_skus)) &
        (w_sku["week_start"] >= min_w52)
    ].sort_values(["StockCode", "week_start"])

    sku_history = {}
    for sku, grp in w_sku_52.groupby("StockCode"):
        sku_code = str(sku)
        series = []
        for _, r in grp.iterrows():
            q = float(r["retail_qty"])
            series.append({
                "week_start": str(r["week_start"])[:10],
                "qty": int(q) if q.is_integer() else round(q, 1)
            })
        sku_history[sku_code] = series

    # Guarantee all 1,760 universe SKUs are present
    for sku in universe_skus:
        s_code = str(sku)
        if s_code not in sku_history:
            sku_history[s_code] = []

    history_path = out_dir / "sku_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(sku_history, f, separators=(",", ":"))
    print(f"Saved: {history_path} ({len(sku_history):,} SKUs)")

    # -------------------------------------------------------------
    # 5. Build File 4: sku_forecast.json (Next 13 Weeks)
    # -------------------------------------------------------------
    print("\n--- 5. BUILDING sku_forecast.json ---")
    fc_13_sorted = fc_13[fc_13["StockCode"].isin(universe_skus)].sort_values(["StockCode", "week_ahead"])

    sku_forecast = {}
    for sku, grp in fc_13_sorted.groupby("StockCode"):
        sku_code = str(sku)
        fc_list = []
        for _, r in grp.iterrows():
            pt = float(r["point_forecast"])
            p10_v = float(r["p10"])
            p90_v = float(r["p90"])
            fc_list.append({
                "week_start": str(r["week_start"])[:10],
                "week_ahead": int(r["week_ahead"]),
                "point_forecast": round(pt, 1),
                "p10": round(p10_v, 1),
                "p90": round(p90_v, 1)
            })
        sku_forecast[sku_code] = fc_list

    # Guarantee all 1,760 universe SKUs are present
    for sku in universe_skus:
        s_code = str(sku)
        if s_code not in sku_forecast:
            sku_forecast[s_code] = []

    forecast_path = out_dir / "sku_forecast.json"
    with open(forecast_path, "w", encoding="utf-8") as f:
        json.dump(sku_forecast, f, separators=(",", ":"))
    print(f"Saved: {forecast_path} ({len(sku_forecast):,} SKUs)")

    # -------------------------------------------------------------
    # 6. Build File 5: inventory_policy.json
    # -------------------------------------------------------------
    print("\n--- 6. BUILDING inventory_policy.json ---")
    inv_records = []
    for _, r in inv.sort_values("StockCode").iterrows():
        s_code = str(r["StockCode"])
        if s_code not in universe_skus:
            continue
        ss_q = float(r["ss_quantile"])
        ss_c95 = float(r["ss_classical_95"])
        ss_c90 = float(r["ss_classical_90"])
        abs_diff = abs(ss_q - ss_c95)
        denom = max(ss_c95, 1.0)
        rel_diff = abs_diff / denom
        disagree_flag = bool(rel_diff > 0.50)

        inv_records.append({
            "stock_code": s_code,
            "abc_class": str(r["abc_class"]),
            "lead_time_weeks": int(r["lead_time_weeks"]),
            "mu_d": round(float(r["mu_d"]), 1),
            "ss_classical_90": round(ss_c90, 1),
            "ss_classical_95": round(ss_c95, 1),
            "ss_quantile": round(ss_q, 1),
            "rop_recommended": round(float(r["rop_recommended"]), 1),
            "disagreement_flag": disagree_flag,
            "disagreement_pct": round(float(rel_diff * 100.0), 1)
        })

    policy_path = out_dir / "inventory_policy.json"
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(inv_records, f, separators=(",", ":"))
    print(f"Saved: {policy_path} ({len(inv_records):,} SKUs)")

    # -------------------------------------------------------------
    # 7. Build File 6: model_performance.json
    # -------------------------------------------------------------
    print("\n--- 7. BUILDING model_performance.json ---")
    model_perf = {
        "sku_leaderboard": [
            {"model": "MA4 (Selected Point Engine)", "wape": 75.83, "bias": -17.42, "mase": 0.940, "is_best_wape": True},
            {"model": "LightGBM (Tweedie)", "wape": 76.28, "bias": -4.19, "mase": 0.919, "is_best_bias": True, "is_best_mase": True},
            {"model": "MA13", "wape": 76.91, "bias": -13.92, "mase": 0.985},
            {"model": "TSB", "wape": 77.05, "bias": -8.16, "mase": 0.986},
            {"model": "Croston-SBA", "wape": 78.15, "bias": -9.77, "mase": 1.023},
            {"model": "SES", "wape": 81.72, "bias": -31.10, "mase": 0.994},
            {"model": "LightGBM (P50 Quantile)", "wape": 68.98, "bias": -39.05, "mase": 0.753, "note": "Median objective; not valid for aggregate volume"}
        ],
        "macro_by_fold": [
            {"fold": 1, "model": "LightGBM (Tweedie)", "macro_wape": 18.70, "macro_bias": -8.93, "winner": False},
            {"fold": 1, "model": "SES_seasonal", "macro_wape": 16.23, "macro_bias": -13.91, "winner": True},
            {"fold": 2, "model": "LightGBM (Tweedie)", "macro_wape": 11.52, "macro_bias": 10.16, "winner": False},
            {"fold": 2, "model": "MA4", "macro_wape": 8.44, "macro_bias": -3.20, "winner": True},
            {"fold": 3, "model": "LightGBM (Tweedie)", "macro_wape": 16.19, "macro_bias": -10.45, "winner": False},
            {"fold": 3, "model": "SES_seasonal", "macro_wape": 9.04, "macro_bias": -5.87, "winner": True}
        ],
        "policy_simulation": [
            {"policy": "Quantile ROP", "fill_rate": 79.20, "stockout_pct": 11.03, "unfulfilled_pct": 20.80, "avg_overstock": 217960, "target": "90% Empirical Non-Parametric"},
            {"policy": "Classical 90% CSL", "fill_rate": 83.35, "stockout_pct": 13.89, "unfulfilled_pct": 16.65, "avg_overstock": 194147, "target": "90% Parametric Gaussian (z=1.282)"},
            {"policy": "Classical 95% CSL", "fill_rate": 87.53, "stockout_pct": 10.63, "unfulfilled_pct": 12.47, "avg_overstock": 233892, "target": "95% Parametric Gaussian (z=1.645)"},
            {"policy": "Naive Benchmark (Hold 4 Wks)", "fill_rate": 70.31, "stockout_pct": 19.75, "unfulfilled_pct": 29.69, "avg_overstock": 148318, "target": "Fixed 4-Week Blanket Demand"}
        ]
    }

    perf_path = out_dir / "model_performance.json"
    with open(perf_path, "w", encoding="utf-8") as f:
        json.dump(model_perf, f, separators=(",", ":"))
    print(f"Saved: {perf_path}")

    # -------------------------------------------------------------
    # 8. Build File 7: key_findings.json
    # -------------------------------------------------------------
    print("\n--- 8. BUILDING key_findings.json ---")
    findings_data = {
        "findings": [
            "Demand Intermittency & Non-Normality: 36.3% of active SKUs exhibit intermittent or lumpy demand patterns (ADI > 1.32). Standard Gaussian safety stock formulas assume symmetric bell-curve error distributions, leading to severe over-buffering on slow movers and dangerous stockouts on surge lines.",
            "Wholesale Demand Contamination: Segregating wholesale bulk orders (>40% of SKU volume and 3x median weekly sales) improved forecasting accuracy across every single baseline model (MA4 improved by 1.36 pp; Seasonal Naive by 3.47 pp). Large non-repeating B2B contracts must be planned out-of-band.",
            "The Macro Aggregation Bias Paradox: While LightGBM P50 achieves lower individual SKU WAPE (68.98%), its median objective carries a -39.05% aggregate bias on zero-inflated, right-skewed demand. For macro warehouse capacity planning, Poisson/Tweedie regression (-4.19% bias) or seasonal moving averages are mathematically essential.",
            "Production Model Selection (MA4): A 4-week Moving Average (MA4) on retail demand achieved 75.83% WAPE, virtually tying complex LightGBM Tweedie (76.28% WAPE) with zero retraining overhead, instant SQL execution, and total operational transparency.",
            "Calibrated Uncertainty Quantification: LightGBM quantile regression models (P10, P90) achieved 80.7% empirical coverage across rolling backtest folds, providing dynamic, highly calibrated lead-time uncertainty bands for safety stock sizing.",
            "The 50.1% Safety Stock Divergence: 882 of 1,760 SKUs (50.1%) exhibit >50% relative difference between classical Gaussian and empirical quantile safety stock. Classical formulas over-buffer zero-inflated Class B lines while under-buffering seasonal surge lines.",
            "Simulation Proof & Operational Tradeoffs: In a 13-week Q4 peak simulation (Fold 3, 1.31M units), Quantile ROP slashed stockout SKU-weeks from 19.75% (naive rule) down to 11.03%, performing comparably to Classical 90% CSL (13.89% stockouts). Quantile ROP is recommended structurally because it eliminates parametric Gaussian assumptions.",
            "ERP Deployment & Replenishment Workflow: Production implementation utilizes recommended Reorder Points (ROP) directly in ERP purchase orders, triggering replenishment whenever inventory position falls below ROP, with rolling 4-week forecast parameter refreshes."
        ]
    }

    findings_path = out_dir / "key_findings.json"
    with open(findings_path, "w", encoding="utf-8") as f:
        json.dump(findings_data, f, separators=(",", ":"))
    print(f"Saved: {findings_path}")

    # Mirror files to outputs/dashboard_data/ for backward compatibility
    json_files = [
        "overview.json",
        "sku_catalog.json",
        "sku_history.json",
        "sku_forecast.json",
        "inventory_policy.json",
        "model_performance.json",
        "key_findings.json"
    ]
    for fname in json_files:
        shutil.copy2(out_dir / fname, mirror_dir / fname)

    # -------------------------------------------------------------
    # 9. Validation: Key Consistency & File Sizes
    # -------------------------------------------------------------
    print("\n--- 9. VALIDATION & CONSISTENCY CHECK ---")
    catalog_skus = set(item["stock_code"] for item in catalog)
    history_skus = set(sku_history.keys())
    forecast_skus = set(sku_forecast.keys())
    policy_skus = set(item["stock_code"] for item in inv_records)

    print(f"Universe SKUs count:         {len(universe_skus):,}")
    print(f"sku_catalog.json SKUs count: {len(catalog_skus):,}")
    print(f"sku_history.json SKUs count: {len(history_skus):,}")
    print(f"sku_forecast.json SKUs count:{len(forecast_skus):,}")
    print(f"inventory_policy.json count: {len(policy_skus):,}")

    assert catalog_skus == universe_skus, "Discrepancy in catalog SKUs!"
    assert history_skus == universe_skus, "Discrepancy in history SKUs!"
    assert forecast_skus == universe_skus, "Discrepancy in forecast SKUs!"
    assert policy_skus == universe_skus, "Discrepancy in inventory policy SKUs!"
    print("\n✅ VALIDATION PASSED: All 4 SKU datasets contain the EXACT same set of 1,760 stock_codes!")

    # File size measurement & 1MB threshold flag
    print("\n--- FILE SIZES REPORT (outputs/dashboard/data/) ---")
    flagged_1mb = []
    tot_bytes = 0
    for fname in json_files:
        fpath = out_dir / fname
        fsize = fpath.stat().st_size
        tot_bytes += fsize
        size_kb = fsize / 1024
        size_mb = fsize / (1024 * 1024)
        status = "OK (< 1MB)"
        if fsize > 1 * 1024 * 1024:
            status = "⚠️ FLAGGED (> 1MB)"
            flagged_1mb.append((fname, size_mb))
        print(f"{fname:<25}: {fsize:>10,} bytes | {size_kb:>8.2f} KB | {size_mb:>6.2f} MB | {status}")

    tot_mb = tot_bytes / (1024 * 1024)
    print(f"\nTotal Uncompressed Data Size: {tot_bytes:,} bytes ({tot_mb:.2f} MB)")
    if flagged_1mb:
        print(f"Flagged Files (> 1MB): {flagged_1mb}")
    else:
        print("All files are under 1MB.")

    # -------------------------------------------------------------
    # 10. Zip Archive Creation (in outputs/dashboard_data/ and outputs/dashboard/)
    # -------------------------------------------------------------
    for z_target in [mirror_dir / "dashboard_data.zip", Path("outputs/dashboard/dashboard_data.zip")]:
        with zipfile.ZipFile(z_target, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for fname in json_files:
                fpath = out_dir / fname
                zipf.write(fpath, arcname=fname)
        print(f"Created Zip File: {z_target} ({z_target.stat().st_size / 1024:.2f} KB)")

    print(f"\nDashboard data export completed successfully in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    export_dashboard_data()
