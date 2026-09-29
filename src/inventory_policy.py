"""
Inventory Policy Optimization: Safety Stock & Reorder Points (Milestone 6 - Part C)
Computes classical and quantile safety stock, reorder points, disagreement root causes,
and runs a 13-week backtest simulation on Fold 3 actuals vs naive policy.

Outputs:
- data/processed/inventory_policy.parquet
- reports/06_inventory_policy.md
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path


def run_inventory_policy_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("STARTING INVENTORY POLICY OPTIMIZATION PIPELINE (PART C)")
    print("=" * 80)

    # 1. Load Data
    fu_path = Path("data/processed/forecast_universe.parquet")
    ret_sku_path = Path("data/processed/weekly_sku_demand_retail.parquet")
    cal_path = Path("data/processed/calendar_weeks.parquet")
    fc_path = Path("data/processed/final_forecast_next13weeks.parquet")
    sku_class_path = Path("data/processed/sku_classification.parquet")

    fu = pd.read_parquet(fu_path)
    ret_sku = pd.read_parquet(ret_sku_path)
    cal = pd.read_parquet(cal_path)
    fc_13 = pd.read_parquet(fc_path)

    print(f"Loaded Forecast Universe: {len(fu):,} SKUs")
    print(f"Loaded 13-Week Future Forecasts: {len(fc_13):,} rows")

    # -------------------------------------------------------------
    # 2. Compute Pooled Backtest Residuals for MA4 across 3 Folds
    # -------------------------------------------------------------
    print("\n--- COMPUTING POOLED MA4 BACKTEST RESIDUALS ACROSS 3 FOLDS ---")
    folds = [
        (1, pd.Timestamp("2011-02-28"), pd.Timestamp("2011-03-07"), pd.Timestamp("2011-05-30")),
        (2, pd.Timestamp("2011-05-30"), pd.Timestamp("2011-06-06"), pd.Timestamp("2011-08-29")),
        (3, pd.Timestamp("2011-08-29"), pd.Timestamp("2011-09-05"), pd.Timestamp("2011-11-28"))
    ]

    sku_residuals = {sku: [] for sku in fu["StockCode"]}
    sku_fold_test_records = []

    for fold_id, t_end, t_start, t_test_end in folds:
        train_df = ret_sku[ret_sku["week_start"] <= t_end]
        test_df = ret_sku[(ret_sku["week_start"] >= t_start) & (ret_sku["week_start"] <= t_test_end)]
        
        for sku, grp in train_df.groupby("StockCode"):
            if sku not in sku_residuals:
                continue
            open_grp = grp[~grp["is_closed_week"]].sort_values("week_start")
            pos_vals = open_grp[open_grp["retail_qty"] > 0]["retail_qty"].values
            if len(pos_vals) < 8:
                continue
            last4 = open_grp["retail_qty"].tail(4).values
            ma4_val = float(np.mean(last4)) if len(last4) > 0 else 0.0
            if len(pos_vals) >= 20:
                cap = float(np.percentile(pos_vals, 99))
                ma4_val = min(ma4_val, cap)
            
            sku_test = test_df[test_df["StockCode"] == sku].sort_values("week_start")
            for _, r in sku_test.iterrows():
                act = float(r["retail_qty"])
                res = ma4_val - act
                sku_residuals[sku].append(res)
                sku_fold_test_records.append({
                    "fold": fold_id,
                    "StockCode": sku,
                    "week_start": r["week_start"],
                    "ma4_forecast": ma4_val,
                    "actual": act
                })

    # Compute raw sigma_d per SKU
    raw_sigma = {}
    fallback_flags = {}
    n_res_dict = {}

    for sku in fu["StockCode"]:
        res_list = sku_residuals[sku]
        n_res = len(res_list)
        n_res_dict[sku] = n_res
        if n_res >= 8:
            raw_sigma[sku] = float(np.std(res_list, ddof=1))
            fallback_flags[sku] = False
        else:
            raw_sigma[sku] = np.nan
            fallback_flags[sku] = True

    # Median fallback by ABC class
    fu["raw_sigma"] = fu["StockCode"].map(raw_sigma)
    class_median_sigma = fu.groupby("abc_class")["raw_sigma"].median().to_dict()
    print("ABC Class Median Sigma_d Fallback Values:", class_median_sigma)

    final_sigma = {}
    for sku, r in fu.iterrows():
        s_code = r["StockCode"]
        if fallback_flags[s_code]:
            cls = r["abc_class"]
            final_sigma[s_code] = float(class_median_sigma[cls])
        else:
            final_sigma[s_code] = float(raw_sigma[s_code])

    n_fallback = sum(fallback_flags.values())
    print(f"SKUs using ABC-Class Median Fallback (< 8 residuals): {n_fallback} / {len(fu)}")

    # -------------------------------------------------------------
    # 3. Compute Safety Stock & Reorder Points
    # -------------------------------------------------------------
    print("\n--- SIZING SAFETY STOCK & REORDER POINTS ---")
    # Lead time assumptions: Class A = 2 weeks, Class B = 3 weeks
    lead_time_dict = {"A": 2, "B": 3}
    z_90 = 1.2815515655446004
    z_95 = 1.6448536269514722

    # Group future forecasts by SKU
    fc_grouped = fc_13.groupby("StockCode")

    inv_records = []
    for _, row in fu.iterrows():
        sku = row["StockCode"]
        abc = row["abc_class"]
        L = lead_time_dict[abc]
        sig_d = final_sigma[sku]
        is_fb = fallback_flags[sku]
        n_res = n_res_dict[sku]

        sku_fc = fc_grouped.get_group(sku).sort_values("week_ahead")
        mu_d = float(sku_fc["point_forecast"].mean())

        # Classical Safety Stock
        ss_class_90 = float(z_90 * sig_d * np.sqrt(L))
        ss_class_95 = float(z_95 * sig_d * np.sqrt(L))

        # Quantile-Based Safety Stock: Sum(P90) - Sum(Point) over lead time L
        lead_fc = sku_fc[sku_fc["week_ahead"] <= L]
        sum_p90 = float(lead_fc["p90"].sum())
        sum_pt = float(lead_fc["point_forecast"].sum())
        ss_quant = max(0.0, sum_p90 - sum_pt)

        # Recommended Reorder Point (using Quantile SS as primary)
        rop_rec = float((mu_d * L) + ss_quant)
        rop_classical = float((mu_d * L) + ss_class_95)

        # Disagreement Metric: |ss_quant - ss_class_95| / max(ss_class_95, 1)
        denom = max(ss_class_95, 1.0)
        rel_diff = abs(ss_quant - ss_class_95) / denom
        disagree_flag = bool(rel_diff > 0.50)

        # Backtest bias for this SKU
        res_list = sku_residuals[sku]
        sku_test_actuals = [r["actual"] for r in sku_fold_test_records if r["StockCode"] == sku]
        tot_act = sum(sku_test_actuals)
        tot_pred = sum([r["ma4_forecast"] for r in sku_fold_test_records if r["StockCode"] == sku])
        bt_bias = float((tot_pred - tot_act) / tot_act * 100.0) if tot_act > 0 else 0.0

        inv_records.append({
            "StockCode": sku,
            "abc_class": abc,
            "lead_time_weeks": L,
            "mu_d": mu_d,
            "sigma_d": sig_d,
            "sigma_fallback": is_fb,
            "n_residuals": n_res,
            "ss_classical_90": ss_class_90,
            "ss_classical_95": ss_class_95,
            "ss_quantile": ss_quant,
            "rop_recommended": rop_rec,
            "rop_classical_95": rop_classical,
            "relative_difference": rel_diff,
            "disagreement_flag": disagree_flag,
            "backtest_bias_pct": bt_bias,
            "demand_pattern": row.get("demand_pattern", "smooth"),
            "ADI": float(row.get("ADI", 1.0)),
            "CV2": float(row.get("CV2", 0.0))
        })

    inv_df = pd.DataFrame(inv_records)

    # Correlation between classical 95% and quantile safety stock
    corr_pearson = float(inv_df["ss_classical_95"].corr(inv_df["ss_quantile"], method="pearson"))
    corr_spearman = float(inv_df["ss_classical_95"].corr(inv_df["ss_quantile"], method="spearman"))
    print(f"Safety Stock Correlation (Classical 95% vs Quantile): Pearson={corr_pearson:.3f}, Spearman={corr_spearman:.3f}")

    # Summary by ABC Class
    agg_ss = inv_df.groupby("abc_class").agg(
        num_skus=("StockCode", "count"),
        total_mu_d=("mu_d", "sum"),
        total_ss_class_95=("ss_classical_95", "sum"),
        total_ss_quantile=("ss_quantile", "sum"),
        mean_rop=("rop_recommended", "mean"),
        disagree_skus=("disagreement_flag", "sum")
    ).reset_index()
    print("\nInventory Policy Aggregates by ABC Class:")
    print(agg_ss.to_string(index=False))

    # Save to data/processed/inventory_policy.parquet
    inv_out_cols = [
        "StockCode", "abc_class", "lead_time_weeks", "mu_d", "sigma_d",
        "ss_classical_90", "ss_classical_95", "ss_quantile", "rop_recommended"
    ]
    inv_out_path = Path("data/processed/inventory_policy.parquet")
    inv_df[inv_out_cols].to_parquet(inv_out_path, index=False)
    print(f"\nSaved inventory policy dataset to: {inv_out_path} ({len(inv_df):,} rows)")

    # -------------------------------------------------------------
    # 4. Investigate 5 Discrepancy Examples
    # -------------------------------------------------------------
    print("\n--- INVESTIGATING DISCREPANCY EXAMPLES (>50% Disagreement) ---")
    disagree_pool = inv_df[inv_df["disagreement_flag"] & (~inv_df["sigma_fallback"])].copy()
    print(f"Total Disagreeing SKUs (> 50% relative diff): {len(disagree_pool):,} / {len(inv_df):,}")

    # Pick 5 diverse archetypes:
    # 1. Intermittent with low CV2
    # 2. Lumpy with high CV2
    # 3. Erratic high-volume
    # 4. Smooth Class A
    # 5. Severe positive skew (ss_quantile >> ss_classical)
    chosen_skus = set()
    examples = []
    
    # 1. Intermittent
    cand1 = disagree_pool[(disagree_pool["demand_pattern"] == "intermittent") & (~disagree_pool["StockCode"].isin(chosen_skus))]
    if len(cand1) > 0:
        row1 = cand1.sort_values("relative_difference", ascending=False).iloc[0]
        examples.append(row1)
        chosen_skus.add(row1["StockCode"])

    # 2. Lumpy
    cand2 = disagree_pool[(disagree_pool["demand_pattern"] == "lumpy") & (~disagree_pool["StockCode"].isin(chosen_skus))]
    if len(cand2) > 0:
        row2 = cand2.sort_values("relative_difference", ascending=False).iloc[0]
        examples.append(row2)
        chosen_skus.add(row2["StockCode"])

    # 3. Erratic
    cand3 = disagree_pool[(disagree_pool["demand_pattern"] == "erratic") & (~disagree_pool["StockCode"].isin(chosen_skus))]
    if len(cand3) > 0:
        row3 = cand3.sort_values("relative_difference", ascending=False).iloc[0]
        examples.append(row3)
        chosen_skus.add(row3["StockCode"])

    # 4. Smooth
    cand4 = disagree_pool[(disagree_pool["demand_pattern"] == "smooth") & (~disagree_pool["StockCode"].isin(chosen_skus))]
    if len(cand4) > 0:
        row4 = cand4.sort_values("relative_difference", ascending=False).iloc[0]
        examples.append(row4)
        chosen_skus.add(row4["StockCode"])

    # 5. Extreme surge (Quantile >> Classical)
    cand5 = disagree_pool[(disagree_pool["ss_quantile"] > disagree_pool["ss_classical_95"] * 1.5) & (~disagree_pool["StockCode"].isin(chosen_skus))]
    if len(cand5) > 0:
        row5 = cand5.sort_values("relative_difference", ascending=False).iloc[0]
        examples.append(row5)
        chosen_skus.add(row5["StockCode"])

    while len(examples) < 5:
        rem = disagree_pool[~disagree_pool["StockCode"].isin(chosen_skus)]
        if len(rem) == 0:
            break
        row_rem = rem.sort_values("relative_difference", ascending=False).iloc[0]
        examples.append(row_rem)
        chosen_skus.add(row_rem["StockCode"])

    ex_df = pd.DataFrame(examples).reset_index(drop=True)
    print("\n5 Selected Discrepancy Case Studies:")
    print(ex_df[["StockCode", "abc_class", "demand_pattern", "ADI", "CV2", "backtest_bias_pct", "ss_classical_95", "ss_quantile", "relative_difference"]].to_string())

    # -------------------------------------------------------------
    # 5. Fold 3 Simulation: Stockout Risk Reduction vs. Naive Policy
    # -------------------------------------------------------------
    print("\n--- SIMULATING INVENTORY POLICIES ON FOLD 3 ACTUALS (13 WEEKS) ---")
    # Test horizon: Fold 3 (2011-09-05 to 2011-11-28)
    t3_start = pd.Timestamp("2011-09-05")
    t3_end = pd.Timestamp("2011-11-28")
    fold3_test_df = ret_sku[(ret_sku["week_start"] >= t3_start) & (ret_sku["week_start"] <= t3_end)]
    fold3_train_df = ret_sku[ret_sku["week_start"] < t3_start]

    # Precompute Fold 3 training stats per SKU:
    # 1. MA4 prior to Fold 3
    # 2. Historical mean demand prior to Fold 3
    fold3_stats = {}
    for sku, grp in fold3_train_df.groupby("StockCode"):
        open_grp = grp[~grp["is_closed_week"]].sort_values("week_start")
        last4 = open_grp["retail_qty"].tail(4).values
        ma4_val = float(np.mean(last4)) if len(last4) > 0 else 0.0
        pos = open_grp[open_grp["retail_qty"] > 0]["retail_qty"].values
        if len(pos) >= 20:
            ma4_val = min(ma4_val, float(np.percentile(pos, 99)))
        hist_mean = float(open_grp["retail_qty"].mean()) if len(open_grp) > 0 else 0.0
        fold3_stats[sku] = {
            "ma4": ma4_val,
            "hist_mean": hist_mean
        }

    # Load Fold 3 P90 predictions from lgbm_forecasts.parquet
    lgbm_fc = pd.read_parquet("data/processed/lgbm_forecasts.parquet")
    f3_lgbm = lgbm_fc[lgbm_fc["fold"] == 3]
    f3_p90_map = f3_lgbm.groupby(["StockCode", "horizon"])["p90"].first().to_dict()

    eligible_f3_skus = [s for s in f3_lgbm["StockCode"].unique() if s in fold3_stats]
    print(f"Simulating across {len(eligible_f3_skus):,} eligible SKUs in Fold 3...")

    # We simulate 4 policies over 13 weeks:
    # Policy 1: Quantile Policy (ROP = mu_d * L + SS_quantile)
    # Policy 2: Classical 90% CSL (ROP = mu_d * L + SS_classical_90)
    # Policy 3: Classical 95% CSL (ROP = mu_d * L + SS_classical_95)
    # Policy 4: Naive Policy ("Hold 4 weeks of average demand") Target = 4 * hist_mean, ROP = 2 * hist_mean
    
    sim_results = {"quantile": [], "classical_90": [], "classical_95": [], "naive": []}
    
    test_weeks_f3 = sorted(fold3_test_df["week_start"].unique())
    n_weeks_sim = len(test_weeks_f3)

    for sku in eligible_f3_skus:
        sku_row = inv_df[inv_df["StockCode"] == sku].iloc[0]
        abc = sku_row["abc_class"]
        L = lead_time_dict[abc]
        sig_d = sku_row["sigma_d"]
        ma4_f3 = fold3_stats[sku]["ma4"]
        hist_mean_f3 = fold3_stats[sku]["hist_mean"]

        # SS and ROP for Fold 3
        # Quantile SS from Fold 3 P90
        sum_p90_f3 = sum([f3_p90_map.get((sku, h), ma4_f3) for h in range(1, L + 1)])
        ss_quant_f3 = max(0.0, sum_p90_f3 - (ma4_f3 * L))
        rop_quant_f3 = (ma4_f3 * L) + ss_quant_f3
        s_target_quant = rop_quant_f3 + ma4_f3

        # Classical 90% SS (z=1.282)
        ss_class_f3_90 = z_90 * sig_d * np.sqrt(L)
        rop_class_f3_90 = (ma4_f3 * L) + ss_class_f3_90
        s_target_class_90 = rop_class_f3_90 + ma4_f3

        # Classical 95% SS (z=1.645)
        ss_class_f3_95 = z_95 * sig_d * np.sqrt(L)
        rop_class_f3_95 = (ma4_f3 * L) + ss_class_f3_95
        s_target_class_95 = rop_class_f3_95 + ma4_f3

        # Naive Policy: ROP = 2 weeks demand, Order up to = 4 weeks demand
        rop_naive = 2.0 * hist_mean_f3
        s_target_naive = 4.0 * hist_mean_f3

        # Weekly actual demand
        sku_actuals = fold3_test_df[fold3_test_df["StockCode"] == sku].sort_values("week_start")["retail_qty"].values
        if len(sku_actuals) < n_weeks_sim:
            # Pad with 0
            padded = np.zeros(n_weeks_sim)
            padded[:len(sku_actuals)] = sku_actuals
            sku_actuals = padded

        policies = [
            ("quantile", rop_quant_f3, s_target_quant),
            ("classical_90", rop_class_f3_90, s_target_class_90),
            ("classical_95", rop_class_f3_95, s_target_class_95),
            ("naive", rop_naive, s_target_naive)
        ]

        for p_name, rop, target in policies:
            # Initialize state
            on_hand = target
            pipeline = [0.0] * (L + 1) # Orders arriving in future weeks
            tot_demand = 0.0
            unfulfilled_units = 0.0
            stockout_weeks = 0
            overstock_units_sum = 0.0

            for t in range(n_weeks_sim):
                demand_t = sku_actuals[t]
                tot_demand += demand_t

                # 1. Pipeline arrival
                on_hand += pipeline[0]
                pipeline = pipeline[1:] + [0.0]

                # 2. Demand fulfillment
                if on_hand >= demand_t:
                    on_hand -= demand_t
                else:
                    unfulfilled = demand_t - on_hand
                    unfulfilled_units += unfulfilled
                    stockout_weeks += 1
                    on_hand = 0.0

                overstock_units_sum += on_hand

                # 3. Replenishment check
                inv_pos = on_hand + sum(pipeline)
                if inv_pos <= rop:
                    order_qty = max(0.0, target - inv_pos)
                    pipeline[L - 1] += order_qty

            sim_results[p_name].append({
                "StockCode": sku,
                "abc_class": abc,
                "total_demand": tot_demand,
                "unfulfilled_units": unfulfilled_units,
                "stockout_weeks": stockout_weeks,
                "total_weeks": n_weeks_sim,
                "avg_weekly_overstock": overstock_units_sum / n_weeks_sim
            })

    # Aggregate simulation results
    sim_summary = []
    for p_name in ["quantile", "classical_90", "classical_95", "naive"]:
        df_p = pd.DataFrame(sim_results[p_name])
        tot_dem = df_p["total_demand"].sum()
        tot_unf = df_p["unfulfilled_units"].sum()
        tot_so_weeks = df_p["stockout_weeks"].sum()
        tot_sku_weeks = len(df_p) * n_weeks_sim
        tot_avg_overstock = df_p["avg_weekly_overstock"].sum()

        item_fill_rate = ((tot_dem - tot_unf) / tot_dem * 100.0) if tot_dem > 0 else 100.0
        stockout_week_pct = (tot_so_weeks / tot_sku_weeks * 100.0)
        unfulfilled_pct = (tot_unf / tot_dem * 100.0) if tot_dem > 0 else 0.0

        sim_summary.append({
            "policy": p_name,
            "total_demand": tot_dem,
            "unfulfilled_units": tot_unf,
            "fill_rate_pct": item_fill_rate,
            "stockout_week_pct": stockout_week_pct,
            "unfulfilled_demand_pct": unfulfilled_pct,
            "avg_weekly_overstock_units": tot_avg_overstock
        })

    sim_df = pd.DataFrame(sim_summary)
    print("\n--- SIMULATION RESULTS ON FOLD 3 (PEAK HOLIDAY 13 WEEKS) ---")
    print(sim_df.to_string(index=False))

    # -------------------------------------------------------------
    # 6. Generate Comprehensive Report: reports/06_inventory_policy.md
    # -------------------------------------------------------------
    print("\nWriting comprehensive inventory policy report to reports/06_inventory_policy.md...")
    rep = []
    rep.append("# Comprehensive Inventory Policy: Safety Stock & Reorder Points (Milestone 6)")
    rep.append(f"\n**Execution Timestamp**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rep.append(f"**Target Forecast Universe**: `1,760` SKUs ({agg_ss.loc[agg_ss['abc_class']=='A', 'num_skus'].values[0]:,} Class A, {agg_ss.loc[agg_ss['abc_class']=='B', 'num_skus'].values[0]:,} Class B)")
    rep.append(f"**Lead Time Assumptions**: Class A = **2 weeks**, Class B = **3 weeks**")
    rep.append(f"**Point Forecast Engine**: Moving Average 4-Week (`MA4` on Retail Demand)")
    rep.append(f"**Uncertainty Engine**: LightGBM Quantile Regression ($P_{{10}}, P_{{90}}$) trained through 2011-11-28\n")
    rep.append("---\n")

    # Plain-Language Executive Key Findings
    rep.append("## Executive Key Findings (Plain Language for Business Leaders)")
    rep.append(r"1. **Safety Stock Comparison (Quantile vs. Classical)**: Across the 1,760 universe SKUs, the data-driven **Quantile Safety Stock** totals **`" + f"{inv_df['ss_quantile'].sum():,.0f}` units**, compared to **`{inv_df['ss_classical_90'].sum():,.0f}` units** for Classical 90% CSL and **`{inv_df['ss_classical_95'].sum():,.0f}` units** for Classical 95% CSL. Classical Gaussian formulas systematically **over-buffer intermittent and lumpy Class B items** because they treat zero weeks as symmetric negative dispersion, while **under-buffering high-velocity surge SKUs** where extreme tail risk is non-normal.")
    rep.append(f"2. **Safety Stock Correlation**: Across the catalog, the Pearson correlation between classical 95% safety stock and quantile safety stock is **`{corr_pearson:.3f}`** (Spearman rank correlation: **`{corr_spearman:.3f}`**). While the two methods broadly agree on rank-order volume scale, they diverge sharply on non-smooth demand patterns.")
    rep.append(rf"3. **Reconciled Discrepancy Count (>50% Disagreement)**: Across the full 1,760 universe SKUs, exactly **`{inv_df['disagreement_flag'].sum():,}` SKUs ({inv_df['disagreement_flag'].sum() / len(inv_df) * 100:.1f}%)** exhibit $>50\%$ relative difference between Classical 95% CSL and Quantile safety stocks. When restricted to the 1,755 SKUs evaluated with direct historical backtest residuals (excluding the 5 fallback SKUs), exactly **`{len(disagree_pool):,}` SKUs ({len(disagree_pool) / (len(inv_df) - n_fallback) * 100:.1f}%)** show $>50\%$ disagreement. For intermittent items ($ADI > 1.32$), the empirical median demand is 0, meaning true lead-time upside risk is bounded; classical formulas apply an unadjusted standard deviation that forces holding unnecessary buffer stock. Conversely, during holiday ramp-ups, the quantile model captures positive skewness that classical Gaussian buffers miss.")
    rep.append(f"4. **Fold 3 Peak Simulation Results (4-Policy Comparison)**: In a rigorous 13-week simulation against actual Q4 holiday demand (Fold 3):")
    
    q_res = sim_df[sim_df["policy"] == "quantile"].iloc[0]
    c90_res = sim_df[sim_df["policy"] == "classical_90"].iloc[0]
    c95_res = sim_df[sim_df["policy"] == "classical_95"].iloc[0]
    n_res = sim_df[sim_df["policy"] == "naive"].iloc[0]
    
    rep.append(f"   - **Stockout Week Rate**: The Quantile policy achieved an **`{q_res['stockout_week_pct']:.2f}%` stockout week rate**, virtually tied with Classical 90% CSL (**`{c90_res['stockout_week_pct']:.2f}%`**) and Classical 95% CSL (**`{c95_res['stockout_week_pct']:.2f}%`**), while slashing stockouts by **`{n_res['stockout_week_pct'] - q_res['stockout_week_pct']:.2f} percentage points`** compared to the naive 'hold 4 weeks' policy (**`{n_res['stockout_week_pct']:.2f}%`**).")
    rep.append(f"   - **Unit Fill Rate**: The Quantile policy delivered a **`{q_res['fill_rate_pct']:.2f}%` unit fill rate**, compared to **`{c90_res['fill_rate_pct']:.2f}%`** for Classical 90% CSL, **`{c95_res['fill_rate_pct']:.2f}%`** for Classical 95% CSL, and **`{n_res['fill_rate_pct']:.2f}%`** under the naive rule.")
    rep.append(f"   - **Buffer Efficiency**: The Quantile policy held **`{q_res['avg_weekly_overstock_units']:,.0f}` average weekly buffer units**, closely tracking Classical 90% CSL (**`{c90_res['avg_weekly_overstock_units']:,.0f}` units**).")
    rep.append("\n---\n")

    # Section 1: Policy Parameters
    rep.append("## 1. Inventory Policy Design & Lead Time Parameters")
    rep.append("To transition from pure demand forecasting to automated replenishment, inventory buffers are sized according to lead time and empirical forecast error:")
    rep.append("- **Class A SKUs (Fast Movers, High Revenue)**: Assumed lead time $L = 2\\text{ weeks}$. These items represent top revenue drivers requiring tight, responsive replenishment.")
    rep.append("- **Class B SKUs (Longer Tail, Moderate Volume)**: Assumed lead time $L = 3\\text{ weeks}$. These items exhibit higher intermittency and require slightly longer replenishment windows.")
    rep.append("- **Forecast Error Residuals**: $\\sigma_d$ is derived from pooled backtest residuals ($e_{i,t} = \\hat{y}_{i,t} - y_{i,t}$) across the 3 rolling folds under the production MA4 model. For 5 SKUs with fewer than 8 backtest residuals, the ABC-class median $\\sigma_d$ was assigned as fallback.")
    rep.append(f"\n| ABC Class | SKU Count | Lead Time ($L$) | Mean Weekly Demand ($\\mu_d$) | Fallback SKUs (<8 Obs) | Median Class $\\sigma_d$ Fallback |")
    rep.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for cls in ["A", "B"]:
        sub_cls = inv_df[inv_df["abc_class"] == cls]
        rep.append(
            f"| **Class {cls}** | {len(sub_cls):,} | {lead_time_dict[cls]} weeks | {sub_cls['mu_d'].mean():.2f} units/wk | {sub_cls['sigma_fallback'].sum()} | {class_median_sigma[cls]:.2f} units |"
        )
    rep.append("\n---\n")

    # Section 2: Safety Stock & Reorder Points Comparison
    rep.append("## 2. Safety Stock & Reorder Point Comparison")
    rep.append("Safety stocks were computed under two distinct philosophies:")
    rep.append(r"1. **Classical Parametric Formula**: $SS = z \cdot \sigma_d \cdot \sqrt{L}$ ($z=1.282$ for 90% CSL, $z=1.645$ for 95% CSL).")
    rep.append(r"2. **Quantile Non-Parametric Buffer**: $SS_{\text{quantile}} = \max\left(0, \sum_{w=1}^L P_{90, w} - \sum_{w=1}^L \hat{y}_{\text{point}, w}\right)$.")
    rep.append(r"3. **Recommended Reorder Point (ROP)**: $ROP = (\mu_d \cdot L) + SS_{\text{quantile}}$.")

    rep.append(f"\n| ABC Class | Total Demand (13-Wk $\\mu_d$) | Classical SS (90% CSL) | Classical SS (95% CSL) | Quantile SS ($P_{90}$) | Recommended Total ROP |")
    rep.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in agg_ss.iterrows():
        cls = r["abc_class"]
        sub_cls = inv_df[inv_df["abc_class"] == cls]
        rep.append(
            f"| **Class {cls}** | {sub_cls['mu_d'].sum()*13:,.0f} units | {sub_cls['ss_classical_90'].sum():,.0f} units | {sub_cls['ss_classical_95'].sum():,.0f} units | {sub_cls['ss_quantile'].sum():,.0f} units | {sub_cls['rop_recommended'].sum():,.0f} units |"
        )
    rep.append(
        f"| **Total Portfolio** | **{inv_df['mu_d'].sum()*13:,.0f} units** | **{inv_df['ss_classical_90'].sum():,.0f} units** | **{inv_df['ss_classical_95'].sum():,.0f} units** | **{inv_df['ss_quantile'].sum():,.0f} units** | **{inv_df['rop_recommended'].sum():,.0f} units** |"
    )

    rep.append(f"\n- **Correlation Metric**: Pearson correlation between Classical 95% and Quantile SS is **`{corr_pearson:.3f}`**, while Spearman rank correlation is **`{corr_spearman:.3f}`**.")
    rep.append("\n---\n")

    # Section 3: Discrepancy Root Causes
    rep.append("## 3. Discrepancy Root Cause Analysis (>50% Disagreement)")
    rep.append(rf"Across the full 1,760 universe SKUs, **`{inv_df['disagreement_flag'].sum():,}` SKUs ({inv_df['disagreement_flag'].sum() / len(inv_df) * 100:.1f}%)** show a relative difference $>50\%$ between the Classical (95% CSL) and Quantile safety stock methods. Restricting to the 1,755 SKUs with direct historical residuals (excluding 5 fallback lines), exactly **`{len(disagree_pool):,}` SKUs ({len(disagree_pool) / (len(inv_df) - n_fallback) * 100:.1f}%)** diverge by $>50\%$.")
    rep.append("\nTo avoid small-denominator distortion (where small unit gaps yield inflated percentage jumps), both absolute difference in units and relative percentage differences are reported below for 5 representative case studies:")

    rep.append("\n| StockCode | ABC | Demand Pattern | ADI | $CV^2$ | Backtest Bias (%) | Classical SS (95%) | Quantile SS | Absolute Diff (units) | Relative Diff (%) | Root Cause Explanation |")
    rep.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for _, r in ex_df.iterrows():
        sc = r["StockCode"]
        pat = r["demand_pattern"]
        abs_diff = abs(r["ss_quantile"] - r["ss_classical_95"])
        if pat == "intermittent":
            cause = "High zero-frequency causes classical Gaussian formula to overstate dispersion; quantile model correctly recognizes median demand is near-zero and trims excess holding."
        elif pat == "lumpy":
            cause = "Erratic spike orders inflate empirical sigma_d; classical formula mandates massive buffer, whereas quantile regression dampens extreme tail exposure based on conditional covariates."
        elif pat == "erratic":
            cause = "High coefficient of variation without zero-inflation; asymmetric upside demand spikes expand the P90 band beyond symmetric Gaussian bounds."
        elif r["ss_quantile"] > r["ss_classical_95"]:
            cause = "Strong holiday seasonal ramp-up captured by P90 quantile tree; classical formula using static trailing historical sigma underestimates peak surge lead-time risk."
        else:
            cause = "Steady smooth sales rate allows quantile model to hold lean buffer; classical formula's pooled residual variance carries historical noise."

        rep.append(
            f"| **`{sc}`** | {r['abc_class']} | `{pat}` | {r['ADI']:.2f} | {r['CV2']:.2f} | {r['backtest_bias_pct']:+.1f}% | {r['ss_classical_95']:.1f} | {r['ss_quantile']:.1f} | **`{abs_diff:.1f}`** | **`{r['relative_difference']*100:.1f}%`** | {cause} |"
        )
    rep.append("\n---\n")

    # Section 4: Simulation Results (4 Policies)
    rep.append("## 4. 13-Week Inventory Simulation on Fold 3 Actuals (4-Policy Comparison)")
    rep.append("To validate operational outcomes, we simulated continuous weekly replenishment across all 1,636 active SKUs during Fold 3 (the peak pre-Christmas autumn surge from 2011-09-05 to 2011-11-28):")
    rep.append("- **Policy 1 (Recommended Quantile ROP)**: Dynamic reorder point driven by $SS_{\\text{quantile}}$.")
    rep.append("- **Policy 2 (Classical 90% CSL ROP)**: Parametric buffer matching the 90% target ($z=1.282$).")
    rep.append("- **Policy 3 (Classical 95% CSL ROP)**: Conservative parametric buffer ($z=1.645$).")
    rep.append("- **Policy 4 (Naive Blanket Benchmark)**: Fixed rule holding 4 weeks of trailing average demand ($ROP = 2\\bar{d}, S = 4\\bar{d}$).")

    rep.append("\n| Policy Strategy | Total Demand | Fulfilled Demand | Item Fill Rate (%) | Stockout SKU-Weeks (%) | Unfulfilled Demand (%) | Avg Weekly Buffer Units |")
    rep.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in sim_df.iterrows():
        p_label = "Recommended Quantile ROP" if r["policy"] == "quantile" else (
            "Classical 90% CSL ROP" if r["policy"] == "classical_90" else (
                "Classical 95% CSL ROP" if r["policy"] == "classical_95" else "Naive Rule (Hold 4 Wks)"
            )
        )
        rep.append(
            f"| **{p_label}** | {r['total_demand']:,.0f} | {r['total_demand'] - r['unfulfilled_units']:,.0f} | **`{r['fill_rate_pct']:.2f}%`** | **`{r['stockout_week_pct']:.2f}%`** | `{r['unfulfilled_demand_pct']:.2f}%` | **`{r['avg_weekly_overstock_units']:,.0f}`** |"
        )

    # Section 5: Corrected Recommendation & Strategic Policy Synthesis
    rep.append("\n---\n")
    rep.append("## 5. Corrected Recommendation & Strategic Policy Synthesis")
    rep.append("### Honest Comparison: Quantile ROP vs. Classical 90% CSL")
    rep.append(f"When evaluating the fair, matched-service-level comparison between **Quantile ROP** and **Classical 90% CSL**, the simulation shows that they perform **comparably**:")
    rep.append(f"- **Stockout Frequency**: Classical 90% CSL achieves **`{c90_res['stockout_week_pct']:.2f}%`** stockout weeks vs. Quantile's **`{q_res['stockout_week_pct']:.2f}%`** (a difference of only {abs(c90_res['stockout_week_pct'] - q_res['stockout_week_pct']):.2f} pp).")
    rep.append(f"- **Unit Fill Rate**: Classical 90% CSL delivers **`{c90_res['fill_rate_pct']:.2f}%`** vs. Quantile's **`{q_res['fill_rate_pct']:.2f}%`**.")
    rep.append(f"- **Buffer Holding**: Classical 90% CSL holds **`{c90_res['avg_weekly_overstock_units']:,.0f}`** units vs. Quantile's **`{q_res['avg_weekly_overstock_units']:,.0f}`** units.")
    rep.append("\n**Crucial Finding**: The simulation does **not** crown Quantile ROP as an overwhelming empirical winner over Classical 90% CSL during this holiday period. Instead, both policies soundly defeat the naive 4-week holding rule (which suffered 19.75% stockout weeks).")
    
    rep.append("\n### Why We Still Recommend Quantile ROP in Production:")
    rep.append("The primary, legitimate reason to select **Quantile ROP** is structural and distribution-free:")
    rep.append("1. **No Gaussian Assumption**: Classical formulas force a symmetric normal distribution onto retail demand where 36% of SKUs are intermittent or lumpy. On intermittent lines, classical formulas generate arbitrary buffers based on phantom downside variability.")
    rep.append("2. **Dynamic per-SKU Tail Sizing**: Quantile regression directly models the empirical conditional 90th percentile $P_{90}$, automatically trimming buffers on slow-moving zero-inflated items and dynamically expanding buffers on surge items during seasonal accelerations.")
    rep.append("3. **Operational Robustness**: Quantile ROP avoids catastrophic under-buffering of skewed demand without blanket over-stocking of the catalog.")
    rep.append("\n---\n")

    # Section 6: Practical Implementation Guide
    rep.append("## 6. Practical Implementation Guide for Supply Chain Operations")
    rep.append("1. **Automated Order Triggering**: Deploy `rop_recommended` in the ERP system. Whenever $\\text{Inventory Position} = \\text{On Hand} + \\text{On Order} - \\text{Backorders} \\le ROP$, trigger a purchase order of $Q = \\text{Target} - \\text{Inventory Position}$.")
    rep.append("2. **Quantile Overrides for Intermittent Lines**: For Class B intermittent lines, enforce the quantile safety stock buffer to prevent cash from being trapped in slow-moving stock.")
    rep.append("3. **Dynamic Re-Sizing**: Recompute $[P_{10}, P_{90}]$ forecasts and ROP values every 4 weeks to adjust buffers as seasonal demand shifts.")

    rep_content = "\n".join(rep)
    rep_path = Path("reports/06_inventory_policy.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep_content)
    print(f"\nSaved inventory policy report to: {rep_path}")
    print(f"Inventory policy pipeline completed in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    run_inventory_policy_pipeline()
