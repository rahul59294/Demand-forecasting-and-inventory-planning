# ⚡ Autonomous Demand Forecasting & Multi-Echelon Inventory Planning Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Dashboard: Standalone HTML5](https://img.shields.io/badge/Dashboard-HTML5%20%2F%20Chart.js-brightgreen.svg)](outputs/dashboard/index.html)

An end-to-end, production-grade demand forecasting and inventory replenishment architecture evaluated across **104 calendar weeks** and **£18.36M in physical sales** from the UCI Online Retail II dataset.

Combines statistical time-series baselines, wholesale/retail decomposition, global machine learning (LightGBM Tweedie & Quantile regression), and empirical non-parametric safety stock optimization.

---

## 🌟 Key Highlights & Business Results

- **Reconciled Multi-Year Accounting**: Resolved historical unit/revenue discrepancies across 104 contiguous weeks to exact tie-out: **10,436,336 units** and **£18,355,881.69**.
- **Wholesale Shock Isolation**: Filtered 871 high-concentration account-SKU pairs, isolating **175,294 bulk spike units** (2.05% of demand) across 86 affected SKUs to prevent bullwhip inventory distortion.
- **Strict Rolling-Origin Backtesting**: 3 seasonal folds (Spring, Summer, Autumn Peak) across 1,760 Continuing A/B SKUs evaluated against raw, uncapped actuals.
- **Model Leaderboard**:
  - **SKU-Level Point Forecast**: `LightGBM (P50)` achieved lowest WAPE (**68.98%**) and MASE (**0.753**), while `MA4 (Retail)` delivered the best zero-bias operational baseline (**75.83% WAPE**, **-15.22% Bias**).
  - **Macro Point Forecast**: `LightGBM (Tweedie)` delivered near-zero aggregate bias (**-4.19%**) and captured aggregate weekly cashflow with high precision.
- **Dynamic Inventory Replenishment**:
  - Quantile ROP ($P_{90}$) matched classical 90% service level (**99.4% fill rate**, **0 stockout SKU-weeks**).
  - Cut average weekly overstock by **56.4%** (from 1,073,694 to 468,142 units) compared to classical Gaussian formulas that over-buffer zero-inflated intermittency.
- **Bespoke Dribbble-Tier Interactive Dashboard**: Standalone HTML5/CSS3/Chart.js web app running 100% client-side with zero Python/server runtime dependency.

---

## 🖥️ Interactive Web Dashboard

The repository includes a modern, high-craft standalone web dashboard located in [`outputs/dashboard/`](outputs/dashboard/index.html).

### Launching the Dashboard Locally

```bash
cd "outputs/dashboard"
python3 -m http.server 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

### Features
1. **📊 Overview**: Hero Bento KPI cards (£18.36M revenue, 10.44M units, 1,760 universe SKUs), 104-week trend line with shaded Christmas closed weeks, and ABC Pareto revenue donut chart.
2. **🔮 Forecast Explorer**: Autocomplete SKU search across all 1,760 items, dynamic 52-week retail demand history connected to the 13-week point forecast (`MA4`), shaded $P_{10}–P_{90}$ uncertainty fan band, and numerical trajectory table.
3. **📦 Inventory Planner**: Searchable catalog table with multi-filters (ABC class, demand pattern, disagreement flag), disagreement warning badges, and fast client-side pagination.
4. **📈 Model Benchmark**: 13-week Fold 3 peak holiday simulation (grouped bar chart + table comparing Quantile ROP, Classical 90% CSL, Classical 95% CSL, Naive rule), and SKU/macro forecasting leaderboards.
5. **💡 Executive Takeaways**: 8 plain-language executive discovery cards.

---

## 📊 Inventory Policy Comparison (Fold 3 Peak Simulation)

Simulated over the 13-week Q4 Autumn peak horizon (2011-09-05 to 2011-11-28) across 1,636 active SKUs:

| Policy | Target CSL | Fill Rate (%) | Stockout SKU-Wks (%) | Unfulfilled Demand (%) | Avg Weekly Overstock (Units) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Quantile ROP ($P_{90}$)** | **90%** | **99.40%** | **0.00%** | **0.60%** | **468,142** |
| Classical 90% CSL ($z=1.282$) | 90% | 99.40% | 0.00% | 0.60% | 1,073,694 |
| Classical 95% CSL ($z=1.645$) | 95% | 99.41% | 0.00% | 0.59% | 1,228,883 |
| Naive Rule (Hold 4 Wks) | — | 98.70% | 1.10% | 1.30% | 760,250 |

> **Key Takeaway**: Non-parametric Quantile ROP matches the 99.4% fill rate of Classical 90% CSL while eliminating **605,552 units of overstock per week** (-56.4%), avoiding massive working capital lockup.

---

## 📁 Repository Structure

```text
.
├── outputs/
│   ├── dashboard/                  # Standalone HTML5 dashboard
│   │   ├── index.html              # Single self-contained web app (Bespoke UI)
│   │   ├── data/                   # Compact JSON datasets (7 files, ~5.6 MB total)
│   │   ├── vendor/                 # Vendored Chart.js UMD bundle
│   │   ├── dashboard_data.zip      # Compressed data archive (488 KB)
│   │   ├── app.py                  # Streamlit alternative app
│   │   └── requirements.txt        # Streamlit requirements
│   ├── eda/                        # High-resolution exploratory plots
│   └── forecast/                   # Leaderboard & model diagnostic plots
├── reports/
│   ├── 01_data_audit.md            # Raw ingestion audit & anomaly classification
│   ├── 02_cleaning_log.md          # Cancellation matching & returns resolution
│   ├── 03_eda.md                   # Reconciled 104-week exploratory analysis
│   ├── 04_baseline_backtest.md     # 9 time-series baselines across 3 rolling folds
│   ├── 05_lightgbm_and_retail_split.md # Wholesale split & global GBDT models
│   ├── 06_inventory_policy.md      # Multi-policy inventory simulation
│   └── 07_dashboard_readme.md      # Dashboard architecture & GitHub Pages guide
├── src/
│   ├── data_loader.py              # Ingestion & date standardizer
│   ├── data_audit.py               # Integrity & anomaly checks
│   ├── data_cleaning.py            # Cancellation-offsetting & customer reconciliation
│   ├── eda.py                      # ABC Pareto, ADI/CV2 classification, STL decomposition
│   ├── backtest.py                 # Rolling-origin evaluation engine for 9 baselines
│   ├── wholesale_split.py          # Customer concentration & retail filtering
│   ├── lgbm_forecasting.py         # LightGBM Tweedie & Quantile training pipeline
│   ├── final_forecast.py           # Production 13-week operational forecast generator
│   ├── inventory_policy.py         # Classical vs Quantile safety stock & simulation
│   └── export_dashboard_data.py    # Production JSON exporter for client-side web
├── data/
│   ├── raw/                        # Raw UCI Online Retail II dataset (ignored from git)
│   └── processed/                  # Cleaned parquet datasets (16 files, ~24 MB)
├── requirements.txt                # Core Python package dependencies
└── README.md                       # Project documentation
```

---

## 🚀 Reproduction Pipeline

To run the complete data pipeline from end to end:

```bash
# 1. Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Place online_retail_II.csv in data/raw/
# (Download from UCI Machine Learning Repository or Kaggle)

# 3. Execute data audit and cleaning
python3 src/data_audit.py
python3 src/data_cleaning.py

# 4. Exploratory Data Analysis & Classification
python3 src/eda.py

# 5. Baseline Rolling-Origin Backtesting
python3 src/backtest.py

# 6. Wholesale Anomaly Filtering
python3 src/wholesale_split.py

# 7. Global LightGBM Training & Prediction
python3 src/lgbm_forecasting.py

# 8. Production 13-Week Forecast Generation
python3 src/final_forecast.py

# 9. Inventory Policy Sizing & Simulation
python3 src/inventory_policy.py

# 10. Export Compact Web JSON Data
python3 src/export_dashboard_data.py
```

---

## 🛠️ Tech Stack & Methodology

- **Data Processing**: Pandas, NumPy, PyArrow (Parquet-backed columnar storage)
- **Time Series & Statistics**: Statsmodels, SciPy (STL decomposition, Croston-SBA, TSB, SES)
- **Machine Learning**: LightGBM (Tweedie Poisson-Gamma compound distribution, Quantile Pinball loss)
- **Replenishment Theory**: Syntetos-Boylan demand categorization, Classical Safety Stock ($z \times \sigma_L$), Non-parametric Quantile Buffer ($P_{90}$)
- **Frontend / Visualization**: Vanilla JavaScript (ES6+), HTML5, CSS3 Custom Properties, Chart.js 4.4, Streamlit, Matplotlib, Seaborn

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
