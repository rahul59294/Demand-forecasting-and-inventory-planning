# Data Cleaning & Weekly Aggregation Log

**Dataset**: UCI Online Retail II (Cleaned Demand Series)
**Execution Timestamp**: 2026-09-28 21:42:12
**Cleaned Transactions**: `997,639` rows
**Weekly Aggregation Records**: `321,483` SKU-week rows across `4,835` unique physical SKUs

---

## 1. Row-Count Waterfall Table
| Pipeline Stage | Row Count | Rows Dropped / Net | Description |
| :--- | :---: | :---: | :--- |
| **1. Raw uncleaned dataset** | 1,067,371 | -0 | Complete combined dataset (2009-2011) |
| **1a. Drop exact duplicates** | 1,033,036 | -34,335 | Identical rows across all columns |
| **1b. Keep physical SKUs only** | 1,027,056 | -5,980 | Matches ^\d{5}[A-Za-z]{0,3}$ (excludes POST, M, fees, etc.) |
| **1c. Drop reversed sale rows** | 1,021,481 | -5,575 | Dropped 5,575 sales fully offset by matched cancellations |
| **1d. Drop remaining cancellations & non-positives** | 997,639 | -23,842 | Dropped remaining C* cancellations, Quantity <= 0, and Price <= 0 |

---

## 2. Cancellation Pairing Statistics
| Metric | Count | Details |
| :--- | :---: | :--- |
| **Cancellations with Known Customer ID** | 17,586 | Candidate cancellations to match |
| **Matched Cancellations** | 15,769 | Successfully paired with earlier sales of sufficient quantity |
| **Unmatched Cancellations** | 1,817 | No earlier sale found with matching SKU, Customer ID, & qty |
| **Fully Reversed Sales (Dropped)** | 5,575 | Sales where remaining quantity was completely offset |
| **Partially Adjusted Sales (Reduced Qty)** | 9,714 | Sales where remaining quantity was decremented |
| **Distinct Sales Rows Impacted** | 15,211 | $5,575 + 9,714 - 78$ (overlap of sales adjusted then reversed) |
| **Repeat Cancellation Hits on Same Sale** | +558 | 502 sales absorbed multiple cancellation events |
| **Net Reconciled Matched Cancellations** | **15,769** | $15,211 + 558 = 15,769$ (exactly reconciles the 480 difference: $558 - 78 = 480$) |
| **Wholesale Outlier 80,995-unit order (SKU 23843)** | **REMOVED** | Reversed by `C581484` (Confirmed: `True`) |
| **Wholesale Outlier 74,215-unit order (SKU 23166)** | **REMOVED** | Reversed by `C541433` (Confirmed: `True`) |

> **Accounting Reconciliation for the 480 Count Discrepancy**:
> - Matched cancellations count: `15,769` cancellation transactions.
> - Sum of reversed sales (5,575) + adjusted sales (9,714) = `15,289`.
> - **Why they differ by 480**:
>   1. **78 sales rows were first partially reduced and then subsequently fully reversed** on a later return. They were counted in both sets, meaning there are $5,575 + 9,714 - 78 = 15,211$ unique impacted sale rows.
>   2. **502 sales rows absorbed multiple separate cancellation events** (e.g. customer returned items across multiple distinct return transactions), accounting for **558 additional cancellation events** hitting the same original sale row.
>   3. Reconciled identity: $15,211 \text{ unique sales rows} + 558 \text{ repeat cancellations} = 15,769 \text{ matched cancellations}$. Net difference: $558 - 78 = 480$.
>   - The pairing algorithm functioned as designed with zero logic errors.

---

## 3. Weekly SKU Demand Table Overview
- **Total Rows in Weekly Table**: `321,483`
- **Unique Physical SKUs**: `4,835`
- **Calendar Weeks Kept**: `102` complete weeks (`2009-12-07` to `2011-11-28`)
- **Dropped Boundary Weeks**: `2009-11-30` (6 days of data) and `2011-12-05` (5 days of data)
- **Zero-Filled Inactive Weeks**: `128,865` rows (**40.08%** of all SKU-weeks)
- **Non-Zero Demand Weeks**: `192,618` rows (**59.92%** of all SKU-weeks)
- **SKUs with 99th Percentile Capping Applied**: `3,170` SKUs (having $\ge 20$ non-zero demand weeks)

---

## 4. Distribution of Non-Zero Demand Weeks per SKU
| Statistic | Value (Weeks) | Notes |
| :--- | :---: | :--- |
| **Minimum** | 1 week | SKUs that sold in only 1 calendar week |
| **25th Percentile (Q1)** | 12 weeks | Intermittent / low-velocity products |
| **Median (Q2)** | 32 weeks | Median product active sales velocity |
| **75th Percentile (Q3)** | 63 weeks | Frequently ordered repeat products |
| **Maximum** | 102 weeks | Products ordered nearly every single complete week |
| **SKUs with $\ge 52$ Non-Zero Weeks** | **1,603** SKUs | Consistent year-round staple SKUs (**33.2%** of catalog) |

---

## 5. Top 10 Physical SKUs by Total Revenue
| Rank | StockCode | Description | Total Qty Sold | Total Revenue | Median Price | Non-Zero Weeks | Active Weeks |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `22423` | REGENCY CAKESTAND 3 TIER | 24,673 | **£309,455.17** | £12.75 | 89 | 90 |
| 2 | `85123A` | WHITE HANGING HEART T-LIGHT HOLDER | 88,136 | **£241,822.44** | £2.95 | 102 | 104 |
| 3 | `85099B` | JUMBO BAG RED RETROSPOT | 94,014 | **£175,456.80** | £1.95 | 102 | 104 |
| 4 | `47566` | PARTY BUNTING | 27,816 | **£146,462.80** | £4.95 | 101 | 104 |
| 5 | `84879` | ASSORTED COLOUR BIRD ORNAMENT | 76,917 | **£124,333.50** | £1.69 | 102 | 104 |
| 6 | `22086` | PAPER CHAIN KIT 50'S CHRISTMAS | 31,067 | **£105,791.20** | £2.95 | 82 | 104 |
| 7 | `79321` | CHILLI LIGHTS | 15,332 | **£77,788.16** | £4.95 | 95 | 104 |
| 8 | `22386` | JUMBO BAG PINK POLKADOT | 38,854 | **£75,190.88** | £1.95 | 96 | 97 |
| 9 | `22197` | SMALL POPCORN HOLDER | 82,573 | **£74,497.43** | £0.85 | 102 | 104 |
| 10 | `20725` | LUNCH BAG RED RETROSPOT | 38,839 | **£68,795.21** | £1.65 | 102 | 104 |

---

## 6. Automated Pipeline Validation Results
| Validation Check | Status | Verification Detail |
| :--- | :---: | :--- |
| **1. No negative/zero qty, price, or rev** | **PASS** | Verified on all `997,639` clean transactions |
| **2. Unique (StockCode, week_start) pairs** | **PASS** | Zero duplicate keys in `321,483` weekly rows |
| **3. Exact Quantity Reconciliation** | **PASS** | Weekly sum (`10,436,336`) equals clean sales sum (`10,436,336`) |
| **4. Contiguous Weekly Grid per SKU** | **PASS** | Every SKU has zero temporal gaps from first to last week |

---

## 7. Observations & Decisions for Forecasting Stage
1. **Intermittent / Lumpy Demand Dominance**:
   - **40.1%** of weekly observations are zero demand between SKU first and last sales.
   - Only **1,603 out of 4,835 SKUs (33.2%)** have at least 52 weeks of positive demand.
   - *Decision*: Standard ARIMA / Prophet models are well suited for the top staple SKUs (>= 52 weeks), while Croston's Method, SBA (Syntetos-Boylan Approximation), or Poisson/negative binomial GLMs will be needed for intermittent slow-movers.
2. **Unmatched Cancellations (1,817 rows)**:
   - Some cancellations did not match earlier sales because the sale occurred prior to Dec 2009 or under a different guest customer id. These were safely dropped in Step 1d, so they do not contaminate demand.
3. **Guest Transactions (`Customer ID` is null)**:
   - Kept in `clean_transactions` and weekly tables (totaling valid sales volume). In `weekly_sku_demand`, `n_customers` accurately counts distinct known customers.