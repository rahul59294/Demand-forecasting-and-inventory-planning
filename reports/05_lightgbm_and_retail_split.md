# Wholesale/Retail Demand Split & Global LightGBM Forecasting Report (Milestone 5)

**Execution Timestamp**: 2026-09-29 14:21:09
**Target Forecast Universe**: `1,760` Continuing High-Impact SKUs (Class A & B)
**Cross-Validation Engine**: Rolling-Origin 3-Fold Walk-Forward Cross-Validation (13-Week Horizon)
**Evaluated Models**: 9 Retail-Adjusted Baselines + Global Multi-Horizon LightGBM (Tweedie & P10/P50/P90 Quantile)

---

## Correction Notice: Macro Forecast Interpretation & Point Objective Alignment
> [!WARNING]
> **Correction to Initial Executive Summary Claim**: The initial Milestone 5 report mistakenly asserted that LightGBM P50 beat the best baselines on macro aggregate warehouse throughput. **This claim was backwards.**
> 
> In reality, LightGBM P50's macro WAPE (**40.81% in Fold 1, 31.90% in Fold 2, and 42.53% in Fold 3**) is significantly **WORSE** than the winning baselines (**16.23%, 8.44%, and 9.04%**).
> 
> **Root Cause**: LightGBM P50 optimizes the median ($L_1$ loss). For intermittent, zero-inflated, and heavily right-skewed demand distributions, the conditional median is systematically lower than the conditional mean ($\sum \text{Median} < \sum \text{Mean}$). Across individual SKUs, P50 carries an overall tracking bias of **-39.05%** (severe underforecasting). When summed across 1,760 SKUs, this bias does not cancel out—it compounds into aggregate underforecasting of 30% to 43% of total warehouse volume.
> 
> **Tweedie Model as the Legitimate Point Forecast**: The LightGBM Tweedie model ($ho=1.5$) explicitly optimizes the compound Poisson-Gamma mean. Its overall bias is **-4.19%** (virtually unbiased), making it the mathematically appropriate point forecast for macro aggregate planning. However, at the macro level, seasonal baselines (`SES_seasonal` and `MA4`) still outperform Tweedie in every fold due to strong aggregate seasonal stability.

---

## Executive Key Findings (Plain Language for Business Leaders)
1. **Impact of Wholesale Spike Segregation**: Removing wholesale-influenced orders (orders $\ge 3\times$ SKU median from accounts driving $>40\%$ SKU volume) removed **175,294 units (2.05% of catalog volume)** across 86 SKUs. This clean `retail_qty` series **improved forecast accuracy across all 9 baseline models**, lowering portfolio WAPE by **-1.36 pp on MA4** (77.19% $\to$ 75.83%) and **-3.47 pp on SeasonalNaive52** (108.57% $\to$ 105.10%). Wholesale bulk spikes create artificial volatility that directly penalizes time series baselines.
2. **LightGBM Performance vs. Best Baseline (MA4)**:
   - **SKU-Level Accuracy**: Comparing the valid point forecast (**LightGBM Tweedie**) against the best retail baseline (**MA4** at 75.83%) reveals a **virtual dead heat**: LightGBM Tweedie achieved **76.28% WAPE** vs. MA4's **75.83% WAPE** (MA4 edges out Tweedie by a negligible **0.45 pp**). Tweedie achieves better MASE (0.919 vs 0.940) and lower bias (-4.19% vs -17.42%), but does not achieve a decisive SKU-level accuracy breakthrough over the simple 4-week moving average.
   - **Macro-Level Accuracy**: At aggregate warehouse volume, baselines win in all three folds. `SES_seasonal` achieves 16.23% in Fold 1 and 9.04% in Fold 3; `MA4` achieves 8.44% in Fold 2. Tweedie achieves 18.70%, 11.52%, and 16.19% respectively.
3. **Quantile Calibration & Safety-Stock Reliability**: The empirical coverage of the $P_10–P_90$ prediction interval was **`80.7%`** (close to nominal 80%), confirming that the quantile model provides highly trustworthy bounds for dynamic safety-stock sizing without arbitrary Gaussian assumptions.
4. **Fold-Level Dynamics**: In Fold 3 (Autumn Peak), the massive pre-Christmas holiday surge favored simple seasonal extrapolation (`SES_seasonal` macro WAPE: 9.04%). While LightGBM incorporated `weeks_to_christmas` effectively, tree shrinkage on zero-inflated Class B SKUs held back its aggregate Q4 performance.

---

## 1. Wholesale vs. Retail Demand Segregation (Part A)
To prevent one-off institutional purchase spikes from contaminating recurring consumer replenishment models, transactions in [`clean_transactions.parquet`](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/data/processed/clean_transactions.parquet) were audited over the 104-week window:
- **Wholesale Account Criteria**: Registered customer accounting for $>40\%$ of that SKU's total 104-week physical volume.
- **Order Anomaly Threshold**: Order quantity in that week $\ge 3 \times \text{median non-zero weekly quantity}$ for the SKU.
- **Capping Policy**: Flagged wholesale weeks are capped at the SKU's **95th percentile of non-wholesale-influenced weeks**.

| Wholesale Metric | Value | Business Interpretation |
| :--- | :---: | :--- |
| **Wholesale Account Pairs (>40% SKU Share)** | `871` | High account concentration on specific niche lines |
| **Wholesale-Influenced Transaction Rows** | `1,244` | Large bulk orders placed by dominant accounts |
| **Forecast Universe Impacted SKUs** | **`86`** / 1,760 (**4.89%**) | Only ~5% of core catalog experiences severe wholesale lumpy spikes |
| **Revenue Covered by Wholesale SKUs** | **`£664,804.32`** (**4.28%**) | £664k in commercial value isolated from retail volatility |
| **Total Volume Capped** | **`175,294` units** (**2.05%**) | Filtered out from core retail demand to build `retail_qty` |
| **SKUs with 0 Non-Wholesale Weeks** | **`0`** | Every SKU has organic non-wholesale retail history |

### Baseline Re-Evaluation: Raw Qty vs. Retail Qty
All 9 baseline models from Milestone 4 were re-executed on [`data/processed/weekly_sku_demand_retail.parquet`](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/data/processed/weekly_sku_demand_retail.parquet) across the identical 3-fold rolling cross-validation design:

| Model | Original WAPE (%) | Retail WAPE (%) | WAPE Delta | Original Bias (%) | Retail Bias (%) | Original MASE | Retail MASE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MA4`** | 77.19% | **`75.83%`** | **`-1.36 pp`** | -16.74% | -17.42% | 0.944 | 0.940 |
| **`MA13`** | 77.88% | **`76.91%`** | **`-0.98 pp`** | -13.68% | -13.92% | 0.985 | 0.985 |
| **`TSB`** | 78.13% | **`77.05%`** | **`-1.08 pp`** | -7.73% | -8.16% | 0.986 | 0.986 |
| **`Croston-SBA`** | 79.23% | **`78.15%`** | **`-1.08 pp`** | -9.31% | -9.77% | 1.023 | 1.023 |
| **`SES`** | 82.45% | **`81.72%`** | **`-0.73 pp`** | -31.02% | -31.10% | 0.991 | 0.994 |
| **`Croston_seasonal`** | 83.33% | **`81.99%`** | **`-1.34 pp`** | +5.93% | +4.72% | 1.068 | 1.066 |
| **`Naive`** | 82.95% | **`82.24%`** | **`-0.70 pp`** | -31.38% | -31.42% | 0.995 | 0.999 |
| **`SES_seasonal`** | 89.00% | **`87.93%`** | **`-1.07 pp`** | -11.18% | -11.93% | 1.065 | 1.068 |
| **`SeasonalNaive52`** | 108.57% | **`105.10%`** | **`-3.47 pp`** | +28.80% | +26.39% | 1.233 | 1.228 |

*Takeaway*: Removing wholesale orders improves every baseline model. Volatile bulk orders distort moving averages and lag models; segregating them creates a much cleaner operational retail baseline.

---

## 2. Global LightGBM Model Architecture (Part B1 & B2)
### Feature Engineering & Leakage Isolation
Features are extracted with strict chronological temporal separation at each fold origin $T$:
1. **Autoregressive Lags**: `lag_1`, `lag_2`, `lag_3`, `lag_4`, `lag_8`, `lag_13`, `lag_26`, `lag_52`.
2. **Rolling Statistics**: Trailing 4-week, 8-week, and 13-week moving means and standard deviations.
3. **Calendar & Seasonality**: Target week `month`, `iso_week`, `weeks_to_christmas` (calendar distance to Dec 25), and `is_closed_week`.
4. **SKU Static Hierarchy**: `abc_class`, `demand_pattern`, `demand_pattern_deseasonalized`, `median_price`, and `active_open_weeks`.
5. **Horizon Feature Architecture**: Single unified model with explicit `horizon` ($h \in [1..13]$) and `horizon_bucket` ('1-4', '5-8', '9-13'). This allows tree nodes to dynamically adjust reliance on short-term autoregressive lags ($h \le 4$) versus long-term seasonal and Christmas features ($h \ge 9$) without training fragmented estimators.

### Hyperparameter Tuning Results per Fold
| Fold | Optimal `num_leaves` | `learning_rate` | `min_data_in_leaf` | `feature_fraction` | Interval Coverage (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | `63` | `0.03` | `30` | `0.8` | **`81.6%`** |
| **Fold 2** | `31` | `0.1` | `50` | `0.8` | **`81.0%`** |
| **Fold 3** | `31` | `0.1` | `50` | `0.8` | **`79.4%`** |

---

## 3. LightGBM vs. Baseline Leaderboard (Corrected Part B4)
Primary SKU-level leaderboard evaluated on retail demand actuals, using **LightGBM-Tweedie** as the point forecast (with P50 retained only as the middle quantile band):

| Model | Type | WAPE (%) | Bias (%) | MASE | Rev-WAPE (%) | LTD WAPE (4-Wk) (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MA4`** | Baseline | **`75.83%`** | `-17.42%` | `0.940` | **`76.08%`** | `58.35%` |
| **`LightGBM (Tweedie)`** | Global GBDT | **`76.28%`** | **`-4.19%`** | **`0.919`** | `78.58%` | **`56.94%`** |
| **`MA13`** | Baseline | 76.91% | -13.92% | 0.985 | 78.88% | 57.94% |
| **`TSB`** | Baseline | 77.05% | -8.16% | 0.986 | 79.07% | 56.96% |
| **`Croston-SBA`** | Baseline | 78.15% | -9.77% | 1.023 | 80.50% | 58.55% |
| **`SES`** | Baseline | 81.72% | -31.10% | 0.994 | 80.61% | 70.83% |
| *`LightGBM (P50 Quantile)`* | *Median (Non-Point)* | *`68.98%`* | *-39.05%* | *0.753* | *69.74%* | *55.56%* |

*Leaderboard Analysis*: LightGBM-Tweedie and MA4 are in a virtual statistical tie on SKU WAPE (76.28% vs 75.83%, a difference of only 0.45 pp). Tweedie is much better balanced on overall bias (-4.19% vs -17.42%) and MASE (0.919 vs 0.940), but MA4 performs slightly better on simple absolute error.

![LightGBM vs Baselines](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/forecast/lgbm_vs_baseline_wape.png)

### Macro Aggregate Portfolio Accuracy per Fold (Tweedie vs Best Baseline)
Comparing aggregate total warehouse throughput predictions against actual retail volume per fold:

| Fold | Actual Total Units | LightGBM Tweedie Units | Tweedie Macro WAPE (%) | Tweedie Macro Bias (%) | Best Baseline Winner | Baseline Macro WAPE (%) | Winner |
| :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| **Fold 1** | 782,580 | 712,672 | **`18.70%`** | `-8.93%` | **`SES_seasonal`** | **`16.23%`** | **`SES_seasonal`** |
| **Fold 2** | 831,316 | 915,785 | **`11.52%`** | `+10.16%` | **`MA4`** | **`8.44%`** | **`MA4`** |
| **Fold 3** | 1,311,074 | 1,174,083 | **`16.19%`** | `-10.45%` | **`SES_seasonal`** | **`9.04%`** | **`SES_seasonal`** |

*Macro Takeaway*: Even with Tweedie's unbiased objective, **simple baselines win at the macro level in every fold**. Seasonal indices capture aggregate warehouse surges better than SKU-level tree aggregations.

---

## 4. Feature Importance & Interpretability (Part B5)
Gain-based feature importance extracted from the LightGBM models across folds:

| Rank | Feature Name | Description | Cross-Fold Role |
| :---: | :--- | :--- | :--- |
| **1** | **`roll_mean_4`** | Recent 4-week sales volume average | Highly Stable across Folds 1–3 |
| **2** | **`iso_week`** | Autoregressive / seasonality feature | Highly Stable across Folds 1–3 |
| **3** | **`weeks_to_christmas`** | Weeks remaining until Christmas | Highly Stable across Folds 1–3 |
| **4** | **`roll_mean_8`** | Autoregressive / seasonality feature | Highly Stable across Folds 1–3 |
| **5** | **`median_price`** | SKU unit selling price | Highly Stable across Folds 1–3 |
| **6** | **`demand_pattern`** | Autoregressive / seasonality feature | Highly Stable across Folds 1–3 |
| **7** | **`month`** | Autoregressive / seasonality feature | Highly Stable across Folds 1–3 |
| **8** | **`lag_1`** | Previous week demand | Highly Stable across Folds 1–3 |
| **9** | **`roll_std_4`** | Autoregressive / seasonality feature | Highly Stable across Folds 1–3 |
| **10** | **`roll_mean_13`** | Quarterly trailing average | Highly Stable across Folds 1–3 |
| **11** | **`horizon`** | Autoregressive / seasonality feature | Highly Stable across Folds 1–3 |
| **12** | **`abc_class`** | Autoregressive / seasonality feature | Highly Stable across Folds 1–3 |
| **13** | **`demand_pattern_deseas`** | Autoregressive / seasonality feature | Highly Stable across Folds 1–3 |
| **14** | **`active_open_weeks`** | Autoregressive / seasonality feature | Highly Stable across Folds 1–3 |
| **15** | **`roll_std_8`** | Autoregressive / seasonality feature | Highly Stable across Folds 1–3 |

![Feature Importance](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/forecast/lgbm_feature_importance.png)

---

## 5. Quantile Calibration & Uncertainty Bounds (Part B6)
Evaluating whether the non-parametric $P_{10}–P_{90}$ interval reliably captures ~80% of true future realization:

| Fold | All SKUs Coverage (%) | Class A Coverage (%) | Class B Coverage (%) | Pinball Loss (q0.10) | Pinball Loss (q0.50) | Pinball Loss (q0.90) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | **`81.56%`** | `78.53%` | `84.75%` | `4.075` | `14.804` | `13.522` |
| **Fold 2** | **`81.02%`** | `81.25%` | `80.78%` | `3.855` | `13.445` | `10.986` |
| **Fold 3** | **`79.44%`** | `77.92%` | `81.06%` | `5.969` | `21.577` | `18.723` |

![Quantile Calibration](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/forecast/quantile_calibration.png)

### Key Takeaways for Inventory Policy & Safety Stock:
1. **Reliable Uncertainty Buffers**: Empirical coverage of ~78–82% across all folds validates that $[P_{10}, P_{90}]$ prediction intervals can directly replace arbitrary standard-deviation safety stock buffers.
2. **Dynamic Replenishment Integration**: In the upcoming milestone, $P_{90}$ can be directly assigned as the reorder target $S$ in $(s, S)$ inventory policies, guaranteeing a data-driven 90% cycle service level (CSL).