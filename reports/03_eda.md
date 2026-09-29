# Exploratory Data Analysis & Demand Profiling Report (Reconciled)

**Dataset**: UCI Online Retail II (Cleaned 104-Week Series)
**Execution Timestamp**: 2026-09-28 23:02:11
**Analysis Scope**: `4,835` Physical SKUs across `104` Calendar Weeks (`2009-12-07` to `2011-11-28`)
**Reconciled 104-Week Totals**: **`10,436,336` Units** | **`£18,355,881.69` Net Revenue**

---

## Corrections to Previous Findings
This section documents the formal audit and mathematical reconciliation of previous discrepancies, narrative corrections, and methodology enhancements:

### 1. Mathematical Reconciliation of 104-Week Totals
Previously, three disparate totals appeared across the analysis due to conflicting date boundaries and table filter scopes. All tables are now strictly reconciled to the identical 104-week window (`2009-12-07` to `2011-11-28`):

| Metric Source | Previous Figures | Reconciled Figures | Root Cause of Previous Gap |
| :--- | :---: | :---: | :--- |
| **`weekly_sku_demand.parquet`** | 10,436,336 units<br>£18,355,881.69 | **10,436,336 units<br>£18,355,881.69** | **Source of Truth**: Defined strictly on the 104 contiguous calendar weeks. |
| **Monthly Aggregation** | 10,510,457 units<br>£18,480,213.19 | **10,436,336 units<br>£18,355,881.69** | Previously grouped by calendar month `2009-12-01` to `2011-11-30`. That included Dec 1–6, 2009 (+141,667 units, +£254,391.75 from dropped week `2009-11-30`) and excluded Dec 1–4, 2011 (-67,546 units, -£130,060.25 from kept week `2011-11-28`), producing a net gap of **+74,121 units and +£124,331.50**. Now aligned to 52-week cycles. |
| **Customer Concentration** | 10,741,552 units | **10,436,336 units<br>£18,355,881.69** | Previously computed on `clean_transactions` without filtering by the 104 calendar weeks, erroneously including the 2 dropped boundary weeks (`2009-11-30` and `2011-12-05`), which contained **+305,216 units and +£569,458.57**. |

### 2. Business Narrative Correction: Revenue vs. Volume Decomposition
While top-line net revenue grew by **+2.02%** (+£183,785.71), physical unit volume actually declined by **-5.88%** (-315,916 units). Average price realization expanded from **£1.6901/unit** in Year 1 to **£1.8319/unit** in Year 2 (+8.39%).

Decomposing the £183,785.71 revenue increase:
- **Volume Effect**: $(Q_2 - Q_1) \times P_1 = (5,060,210 - 5,376,126) \times £1.6901 =$ **-£533,921.25**
- **Price/Mix Effect**: $Q_2 \times (P_2 - P_1) = 5,060,210 \times (£1.8319 - £1.6901) =$ **+£717,706.96**
- **Net Revenue Impact**: $\Delta \text{Revenue} = \text{Volume Effect} + \text{Price/Mix Effect} =$ **+£183,785.71**

*Takeaway*: Business top-line expansion was entirely driven by price/mix optimization and higher unit prices, masking a underlying volume contraction.

### 3. STL Trend Direction
Evaluating the STL trend component across open operating weeks reveals that the underlying baseline demand softened: mean weekly trend fell from **103,498.1 units/week** across the first 13 open weeks to **92,139.7 units/week** across the last 13 open weeks (a net decline of **-11,358.4 units/week, or -11.0%**).

### 4. Root Cause of Unit Decline (Wholesale Order Concentration in Dec–Apr)
Comparing Year 2 vs Year 1 across the Dec–Apr period, the top 10 SKUs driving the largest unit volume decline were analyzed for customer concentration:

| StockCode | Description | Y1 Qty | Y2 Qty | Unit Decline | Top Customer Drop | Cust Share of Decline | Single Cust > 25% |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `37410` | BLACK AND WHITE PAISLEY FLOW | 25,484 | 0 | **25,484** | Cust 13902 (-25,164) | **98.7%** | **True** |
| `21091` | SET/6 WOODLAND PAPER PLATES | 13,686 | 0 | **13,686** | Cust 13902 (-12,960) | **94.7%** | **True** |
| `21085` | SET/6 WOODLAND PAPER CUPS | 13,644 | 0 | **13,644** | Cust 13902 (-12,744) | **93.4%** | **True** |
| `21099` | SET/6 STRAWBERRY PAPER CUPS | 13,125 | 0 | **13,125** | Cust 13902 (-12,960) | **98.7%** | **True** |
| `21092` | SET/6 STRAWBERRY PAPER PLATE | 12,846 | 0 | **12,846** | Cust 13902 (-12,480) | **97.2%** | **True** |
| `21984` | PACK OF 12 PINK PAISLEY TISS | 14,276 | 1,563 | **12,713** | Cust 17940 (-11,000) | **86.5%** | **True** |
| `21980` | PACK OF 12 RED RETROSPOT TIS | 14,379 | 2,598 | **11,781** | Cust 17940 (-10,000) | **84.9%** | **True** |
| `20993` | JAZZ HEARTS MEMO PAD | 10,966 | 0 | **10,966** | Cust 13902 (-9,312) | **84.9%** | **True** |
| `21981` | PACK OF 12 WOODLAND TISSUES | 12,604 | 2,868 | **9,736** | Cust 17940 (-9,136) | **93.8%** | **True** |
| `21982` | PACK OF 12 SUKI TISSUES | 13,154 | 3,560 | **9,594** | Cust 17940 (-9,072) | **94.6%** | **True** |

> [!IMPORTANT]
> **Wholesale Lumpy Order Findings**: **100% of the top 10 declining SKUs** had over **84% of their volume drop** attributable to a single wholesale account (`Customer 13902` or `Customer 17940`). These accounts placed massive bulk orders in early 2010 that did not repeat in early 2011. The apparent volume loss was not a structural retail demand collapse, but non-repeating B2B bulk purchases.

### 5. Seasonality Metric Caveat & Separate Autumn Shares
- **Caveat on Seasonal Strength ($F_s = 0.9978$)**: The calculated seasonal strength is artificially high because of the two mandatory Christmas shutdown weeks (which have 0 units demanded).
- **Separate Autumn Concentration (Sep–Nov)**: Year 1 = **35.33%** of annual volume; Year 2 = **37.78%** of annual volume, confirming stable seasonal peaking.

---

## Executive Key Findings (For Business Leaders)
- **Extreme Revenue Concentration (Pareto Principle)**: Just **1,044 SKUs (21.6% of catalog)** generate **80.0% of total business revenue** (£14,685,385.52). Focusing predictive modeling on these top items will capture the vast majority of commercial impact.
- **Pervasive Lumpy & Intermittent Demand**: More than **55.9% of the physical catalog** exhibits intermittent or lumpy demand patterns. Only **10.2% of SKUs** exhibit smooth, predictable demand.
- **High Catalog Churn**: **1,591 SKUs (32.9%)** were discontinued prior to the final quarter of 2011, leaving **2,815 active continuing SKUs** for forward forecasting.
- **Customer Account Split**: Registered customer accounts generate **93.7% of physical units** (£15,891,043.57), while guest checkout accounts drive **6.3%** (£2,464,838.12). The top 50 accounts drive **30.4%** of total unit volume.

---

## 1. Calendar Alignment & Closed Weeks (Part A)
A full 104-week calendar (`2009-12-07` to `2011-11-28`) was constructed in [`data/processed/calendar_weeks.parquet`](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/data/processed/calendar_weeks.parquet).

| Week Start | ISO Week | Total Qty | Total Revenue | is_closed_week | Operational Reason |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `2009-12-28` | W53 | 0 | £0.00 | **True** | Annual Christmas/New Year Holiday Shutdown |
| `2010-12-27` | W52 | 0 | £0.00 | **True** | Annual Christmas/New Year Holiday Shutdown |

All interval and rate calculations in this report strictly exclude these 2 closed weeks, operating over the **102 open calendar weeks**.

![Macro Weekly Demand with Shaded Closed Weeks](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/weekly_total_demand.png)

---

## 2. ABC Revenue Segmentation (Part B1)
| ABC Class | SKU Count | % of Catalog | Total Revenue (£) | % Revenue Share | Demand Strategy |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Class A** | 1,044 | 21.59% | £14,685,385.52 | **80.00%** | Priority SKU forecasting (ARIMA / Prophet / ML) |
| **Class B** | 1,265 | 26.16% | £2,753,598.22 | **15.00%** | Standard replenishment models |
| **Class C** | 2,526 | 52.24% | £916,897.95 | **5.00%** | Simple reorder points / rule-based |
| **Total** | **4,835** | **100.00%** | **£18,355,881.69** | **100.00%** | — |

![Pareto ABC Curve](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/eda/pareto_abc.png)

---

## 3. Syntetos-Boylan Demand Pattern Classification (Part B2)
Demand patterns are categorized using Average Demand Interval ($\text{ADI} = \text{active open weeks} / \text{nonzero weeks}$) and squared coefficient of variation ($CV^2 = (\sigma / \mu)^2$ across positive weeks, with `ddof=0`):
- **Smooth** ($\text{ADI} < 1.32$, $CV^2 < 0.49$): Regular frequency, low variability.
- **Erratic** ($\text{ADI} < 1.32$, $CV^2 \ge 0.49$): Regular frequency, high order variability.
- **Intermittent** ($\text{ADI} \ge 1.32$, $CV^2 < 0.49$): Sporadic orders, consistent order size.
- **Lumpy** ($\text{ADI} \ge 1.32$, $CV^2 \ge 0.49$): Sporadic orders, highly variable order size.

| Demand Pattern | SKU Count (Raw) | % Catalog | SKU Count (Deseas) | % Catalog | Recommended Forecasting Approach |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Smooth** | 492 | 10.18% | 482 | 9.97% | Classical ARIMA / ETS / Prophet |
| **Erratic** | 1,639 | 33.90% | 1,649 | 34.11% | LightGBM with lag features & winsorization |
| **Intermittent** | 563 | 11.64% | 524 | 10.84% | Croston's Method / SBA |
| **Lumpy** | 2,141 | 44.28% | 2,180 | 45.09% | Croston's + Poisson / Bootstrapping |
| **Total** | **4,835** | **100.00%** | **4,835** | **100.00%** | — |

### Transition Matrix: Raw vs. Deseasonalized Classification
| Raw Pattern \ Deseasonalized | Smooth | Erratic | Intermittent | Lumpy | Total SKUs |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Smooth** | 446 | 46 | 0 | 0 | 492 |
| **Erratic** | 36 | 1,603 | 0 | 0 | 1,639 |
| **Intermittent** | 0 | 0 | 491 | 72 | 563 |
| **Lumpy** | 0 | 0 | 33 | 2,108 | 2,141 |
| **All** | 482 | 1,649 | 524 | 2,180 | 4,835 |

### ABC × Demand Pattern Matrix (SKU Counts & Revenue Share)
| ABC Class | Smooth (Counts / Rev %) | Erratic (Counts / Rev %) | Intermittent (Counts / Rev %) | Lumpy (Counts / Rev %) | Total SKUs | Total Rev % |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Class A** | 137 (18.1%) | 777 (54.9%) | 6 (0.2%) | 124 (6.7%) | 1,044 | 80.0% |
| **Class B** | 97 (1.2%) | 635 (8.1%) | 13 (0.1%) | 520 (5.6%) | 1,265 | 15.0% |
| **Class C** | 258 (0.2%) | 227 (0.7%) | 544 (0.4%) | 1,497 (3.7%) | 2,526 | 5.0% |

![ADI vs CV2 Scatter Plot](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/eda/adi_cv2_scatter.png)

---

## 4. SKU Lifecycle Status (Part B3)
Relative to the 104-week horizon ending `2011-11-28`:
- **New**: `first_sale_week` in the final 26 weeks (`>= 2011-06-06`).
- **Discontinued**: `last_sale_week` more than 13 weeks before calendar end (`< 2011-08-29`).
- **Continuing**: Sold across historical window and actively reordered in the final quarter.

| Lifecycle Status | SKU Count | % of Catalog | Total Revenue (£) | % Revenue Share | Modeling Treatment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Continuing** | 2,815 | 58.22% | £16,007,601.46 | **87.21%** | Core universe for time-series forecasting |
| **Discontinued** | 1,591 | 32.91% | £1,488,816.81 | **8.11%** | Exclude from future purchase planning |
| **New** | 429 | 8.87% | £859,463.42 | **4.68%** | Cold-start / hierarchy-based forecasting |

### ABC × Lifecycle Matrix (SKU Counts & Revenue Share)
| ABC Class | Continuing (Counts / Rev %) | Discontinued (Counts / Rev %) | New (Counts / Rev %) | Total SKUs | Total Rev % |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Class A** | 901 (74.2%) | 80 (3.2%) | 63 (2.6%) | 1,044 | 80.0% |
| **Class B** | 859 (10.4%) | 271 (3.0%) | 135 (1.6%) | 1,265 | 15.0% |
| **Class C** | 1,055 (2.6%) | 1,240 (1.9%) | 231 (0.5%) | 2,526 | 5.0% |

---

## 5. Seasonality, Monthly Indices & Decomposition (Part B4)
### Monthly Seasonal Index & Year-over-Year Comparison (52-Week Aligned)
| Calendar Month | Seasonal Index | Year 1 Qty (2009-10) | Year 1 Rev (£) | Year 2 Qty (2010-11) | Year 2 Rev (£) | Qty Growth % | Rev Growth % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **December** | **0.880** | 277,898 | £530,139.06 | 262,318 | £581,956.67 | -5.6% | +9.8% |
| **January** | **0.807** | 387,490 | £605,323.45 | 355,484 | £658,483.24 | -8.3% | +8.8% |
| **February** | **0.804** | 374,911 | £526,213.35 | 283,440 | £509,777.56 | -24.4% | -3.1% |
| **March** | **0.949** | 536,769 | £777,354.64 | 336,687 | £604,073.37 | -37.3% | -22.3% |
| **April** | **0.773** | 349,282 | £615,930.71 | 283,762 | £483,957.85 | -18.8% | -21.4% |
| **May** | **0.847** | 435,550 | £715,495.79 | 431,019 | £812,369.12 | -1.0% | +13.5% |
| **June** | **0.861** | 360,552 | £621,628.38 | 344,389 | £611,779.70 | -4.5% | -1.6% |
| **July** | **0.849** | 310,691 | £591,102.54 | 383,936 | £658,400.41 | +23.6% | +11.4% |
| **August** | **0.890** | 443,823 | £726,479.77 | 467,278 | £796,323.42 | +5.3% | +9.6% |
| **September** | **1.292** | 543,493 | £818,906.65 | 513,719 | £932,440.54 | -5.5% | +13.9% |
| **October** | **1.402** | 563,447 | £987,041.82 | 727,590 | £1,291,706.79 | +29.1% | +30.9% |
| **November** | **1.589** | 792,220 | £1,570,431.83 | 670,588 | £1,328,565.03 | -15.4% | -15.4% |
| **Total** | — | **5,376,126** | **£9,086,047.99** | **5,060,210** | **£9,269,833.70** | **-5.9%** | **+2.0%** |

- **Autumn Concentration (Sep–Nov)**: Year 1 = **35.33%** | Year 2 = **37.78%** of annual volume.

![YoY Weekly Overlay](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/eda/yoy_overlay.png)

### STL Time-Series Decomposition Metrics
- **Decomposition Method**: `statsmodels.tsa.seasonal.STL` (period=52, robust=True)
- **Trend Strength ($F_t$)**: **`0.7674`**
- **Seasonal Strength ($F_s$)**: **`0.9978`** *(Note: inflated by the two Christmas closure zero-demand weeks)*
- **Open-Week Trend Shift**: Mean weekly trend decreased from **103,498.1** (first 13 open weeks) to **92,139.7** (last 13 open weeks), representing a **-11,358.4 units/week (-11.0%)** volume softening.

![STL Decomposition Plot](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/eda/stl_decomposition.png)

### Top 6 Class-A SKUs Weekly Demand Behavior
![Top 6 Class A Demand Profiles](file:///Users/rahulkumarsingh/Demand%20Forecasting%20and%20Inventory%20Planning/outputs/eda/top_sku_weekly.png)

---

## 6. Business Growth & Customer Concentration (Part B5)
| Metric | Year 1 (Weeks 0-51) | Year 2 (Weeks 52-103) | YoY Change |
| :--- | :---: | :---: | :---: |
| **Total Units Demanded** | 5,376,126 | 5,060,210 | **-5.88%** |
| **Total Net Sales Revenue** | £9,086,047.99 | £9,269,833.70 | **+2.02%** |
| **Average Realized Price / Unit** | £1.6901 | £1.8319 | **+8.39%** |

### Channel & Account Concentration (104-Week Window)
- **Registered Customer Accounts**: `9,774,145` units (**93.65%**), £15,891,043.57 (**86.57%** of revenue)
- **Guest / Marketplace Transactions (Null Customer ID)**: `662,191` units (**6.35%**), £2,464,838.12 (**13.43%** of revenue)
- **Top 10 Customers Concentration**: `1,615,803` units (**15.48%** of all physical retail volume)
- **Top 50 Customers Concentration**: `3,167,481` units (**30.35%** of all physical retail volume)

---

## 7. Candidate Forecasting Universes (Part B6)
The following candidate universes were evaluated to define the training/evaluation scope for forecasting models:

| Universe Filtering Rule | SKU Count | % of Catalog | Total Revenue Covered (£) | % Revenue Covered | Operational Trade-off |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **(i) ABC Class A & Lifecycle Continuing** | **901** | **18.6%** | **£13,623,382.87** | **74.2%** | Highest revenue density per SKU; captures 74.2% revenue with only 901 SKUs, but leaves secondary B items unforecasted. |
| **(ii) ABC Class A or B & Lifecycle Continuing** | **1,760** | **36.4%** | **£15,533,546.20** | **84.6%** | Balanced operational standard; captures 84.6% revenue across 1,760 SKUs, removing discontinued and inactive churn lines. |
| **(iii) Rule (ii) with >= 40 Non-Zero Weeks** | **1,525** | **31.5%** | **£14,247,987.62** | **77.6%** | Data-rich mature subset; ensures dense history for deep autoregressive lags, but sacrifices recently launched lines. |
| **(iv) Rule (ii) with Smooth or Erratic Pattern** | **1,237** | **25.6%** | **£13,581,443.09** | **74.0%** | Optimized for continuous time-series models (ARIMA/ETS); eliminates zero-order intermittency but drops 36.5% of A/B lines. |