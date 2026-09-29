# Standalone Supply Chain Dashboard: User Guide & GitHub Pages Deployment (Milestone 9)

**Dashboard Path**: `outputs/dashboard/index.html`  
**Data Directory**: `outputs/dashboard/data/`  
**Vendored Libraries**: `outputs/dashboard/vendor/chart.umd.min.js` (204 KB)  
**Deployment Target**: Standalone browser / GitHub Pages / Static Web Server (Zero Python Runtime Required)

---

## 1. Overview & Directory Structure

The **Demand Forecasting & Inventory Planning Dashboard** is a standalone, client-side web application built with vanilla HTML5, CSS3, and vendored [Chart.js](https://www.chartjs.org/). It runs directly in any modern web browser without requiring a Python environment, Node.js server, or external build toolchain.

### Directory Structure of `outputs/dashboard/`

```text
outputs/dashboard/
├── index.html                   # Single self-contained HTML/CSS/JS web application (55 KB)
├── dashboard_data.zip           # Compressed bundle of all 7 JSON files (488 KB)
├── vendor/
│   └── chart.umd.min.js         # Vendored Chart.js UMD bundle for offline execution (204 KB)
├── data/
│   ├── overview.json            # Portfolio totals, closed weeks, and 104-week trend (8.6 KB)
│   ├── sku_catalog.json         # All 1,760 universe SKUs metadata & classification (356 KB)
│   ├── sku_history.json         # Weekly retail actuals for last 52 weeks (3.07 MB)
│   ├── sku_forecast.json        # Next 13 weeks future forecast with P10/P90 bands (1.91 MB)
│   ├── inventory_policy.json    # Replenishment parameters, SS, ROP, disagreement flags (362 KB)
│   ├── model_performance.json   # 4-policy simulation table & SKU/macro leaderboards (1.8 KB)
│   └── key_findings.json        # 8 plain-language executive takeaways (2.3 KB)
├── app.py                       # Streamlit dashboard alternative (Python-based)
└── requirements.txt             # Python dependencies for the Streamlit alternative
```

---

## 2. How to Run Locally

### Why a Local Web Server is Needed
Modern web browsers (Chrome, Firefox, Safari, Edge) enforce strict security policies regarding `file://` URLs, blocking asynchronous `fetch()` requests to local JSON files (`./data/*.json`). Serving the folder through any local HTTP server resolves this instantly.

### Quick Launch Command (Built-in Python HTTP Server)

From the project root, open your terminal and run:

```bash
cd "outputs/dashboard"
python3 -m http.server 8000
```

Then open your browser and navigate to:
```
http://localhost:8000/
```

*(Alternatively, if running from the repository root directory)*:
```bash
python3 -m http.server 8000
# Then visit: http://localhost:8000/outputs/dashboard/
```

### Offline Execution & CDN Fallback
`index.html` references `vendor/chart.umd.min.js` with a relative path. If the local vendored script is ever missing or blocked, the page includes an automatic fallback that asynchronously loads Chart.js from the official jsDelivr CDN.

---

## 3. Step-by-Step GitHub Pages Deployment Guide

Deploying the dashboard to GitHub Pages makes it publicly accessible worldwide with automated SSL certification, global CDN distribution, and zero hosting costs.

### Method 1: Deploying via `gh-pages` Branch (Cleanest Approach)

This method publishes only the dashboard folder to a dedicated `gh-pages` branch:

1. **Commit your changes**:
   ```bash
   git add outputs/dashboard/
   git commit -m "Add standalone GitHub Pages dashboard"
   ```

2. **Push the `outputs/dashboard/` directory to `gh-pages` branch using git subtree**:
   ```bash
   git subtree push --prefix outputs/dashboard origin gh-pages
   ```

3. **Enable GitHub Pages**:
   - Go to your repository on GitHub.
   - Click **Settings** (top tab) &rarr; **Pages** (left sidebar).
   - Under **Build and deployment**:
     - **Source**: Select `Deploy from a branch`.
     - **Branch**: Select `gh-pages` and folder `/ (root)`.
   - Click **Save**.

4. **Access your live dashboard**:
   - Within 1–2 minutes, your dashboard will be live at:
     ```
     https://<your-username>.github.io/<your-repository-name>/
     ```

---

### Method 2: Deploying via `/docs` Folder in the `main` Branch

If you prefer to keep everything in your default `main` branch:

1. **Copy the dashboard contents into a root `/docs` folder**:
   ```bash
   mkdir -p docs
   cp -r outputs/dashboard/* docs/
   git add docs/
   git commit -m "Deploy dashboard to /docs for GitHub Pages"
   git push origin main
   ```

2. **Enable GitHub Pages**:
   - Go to your repository **Settings** &rarr; **Pages**.
   - Under **Build and deployment**:
     - **Source**: Select `Deploy from a branch`.
     - **Branch**: Select `main` and folder `/docs`.
   - Click **Save**.

3. **Access your live dashboard**:
   - Your site will be published at:
     ```
     https://<your-username>.github.io/<your-repository-name>/
     ```

---

## 4. Key Functional Features of the HTML Dashboard

### 1. 📊 Overview
- **Executive Metric Cards**: Total catalog revenue (£18.36M), physical units sold (10.44M), active universe SKUs (1,760), and ABC catalog distribution (1,044 A / 1,265 B / 2,526 C).
- **Interactive 104-Week Total Demand Chart**: High-resolution line chart with custom canvas shading highlighting the two annual holiday closed weeks (`2009-12-28` and `2010-12-27`).
- **ABC Pareto Donut Chart**: Visualizes the 80/20 revenue concentration across catalog items.

### 2. 🔮 Forecast Explorer
- **Autocomplete SKU Selector**: Search by StockCode or product description across all 1,760 universe SKUs.
- **Dynamic Metadata Badges**: Semantic colored tags indicating ABC classification, Syntetos-Boylan demand pattern, lifecycle status, unit price, mean weekly demand ($\mu_d$), quantile safety stock, and recommended reorder point.
- **Uncertainty Trajectory Fan Chart**: Displays 52 weeks of historical retail demand smoothly connected to the 13-week point forecast (`MA4`), enveloped by a shaded $P_{10}–P_{90}$ prediction interval band (80% coverage).
- **Numerical Trajectory Table**: Detailed breakdown showing weekly point forecasts, prediction interval bounds, interval width, and cumulative volume.

### 3. 📦 Inventory Planner
- **Interactive Multi-Filter Bar**:
  - Filter by ABC class (`All`, `A`, `B`).
  - Filter by Demand pattern (`All`, `Smooth`, `Erratic`, `Intermittent`, `Lumpy`).
  - Toggle checkbox to isolate **Disagreeing SKUs (>50% difference)** between classical and quantile buffers.
  - Live text search across StockCodes and product descriptions.
- **Sortable Catalog Table**: Click any column header to sort ascending or descending.
- **Disagreement Diagnostics**: Rows where classical Gaussian formulas diverge from empirical quantiles are highlighted with warning badges and explanatory tooltips.
- **Smooth Client-Side Pagination**: Fast 50-row pagination bar ensuring instantaneous performance across all 1,760 items.

### 4. 📈 Model Performance
- **Peak Holiday Simulation (Fold 3 Actuals)**: Side-by-side grouped bar chart and data table comparing all 4 policies (**Quantile ROP**, **Classical 90% CSL**, **Classical 95% CSL**, and **Naive 4-week holding rule**). Best values in each column are visually highlighted.
- **Forecasting Leaderboards**: Tweedie-corrected SKU WAPE leaderboard and macro aggregate warehouse volume accuracy per fold.

### 5. 💡 Key Findings
- 8 structured executive cards synthesizing takeaways for non-technical supply chain leaders, covering zero-inflation, wholesale order segregation, the macro median bias paradox, MA4 model selection, safety stock divergence, and ERP replenishment integration.

### 6. Design & Accessibility
- **Semantic Color Palette**: Strict semantic mapping where colors carry consistent meaning across all tabs (Class A = Royal Blue, Class B = Amber, Class C = Slate; Smooth = Green, Erratic = Cyan, Intermittent = Orange, Lumpy = Red).
- **Dark / Light Theme Toggle**: Persistent mode switch with CSS custom properties and `localStorage` caching.
- **Fully Responsive**: Adapts seamlessly to laptops, tablets, and mobile screens.

---

## 5. Data Sizing & Validation Audit

```
Universe SKUs count:         1,760
sku_catalog.json SKUs count: 1,760
sku_history.json SKUs count: 1,760
sku_forecast.json SKUs count:1,760
inventory_policy.json count: 1,760

✅ VALIDATION PASSED: All 4 SKU datasets contain the EXACT same set of 1,760 stock_codes!
```

### Data File Sizes in `outputs/dashboard/data/`:

| File Name | Uncompressed Size | Size (KB/MB) | 1MB Threshold Status |
| :--- | :---: | :---: | :--- |
| `overview.json` | 8,835 bytes | 8.63 KB | ✅ OK (< 1 MB) |
| `sku_catalog.json` | 364,446 bytes | 355.90 KB | ✅ OK (< 1 MB) |
| `sku_history.json` | 3,220,517 bytes | **3.07 MB** | ⚠️ **FLAGGED (> 1 MB)** *(52 weekly observations across 1,760 items)* |
| `sku_forecast.json` | 1,999,067 bytes | **1.91 MB** | ⚠️ **FLAGGED (> 1 MB)** *(13-week point, P10, P90 across 1,760 items)* |
| `inventory_policy.json` | 370,221 bytes | 361.54 KB | ✅ OK (< 1 MB) |
| `model_performance.json` | 1,801 bytes | 1.76 KB | ✅ OK (< 1 MB) |
| `key_findings.json` | 2,302 bytes | 2.25 KB | ✅ OK (< 1 MB) |
| **Total Uncompressed** | **5,967,189 bytes** | **5.69 MB** | — |
| **Compressed `dashboard_data.zip`** | **499,671 bytes** | **487.96 KB (0.48 MB)** | **91.6% compression ratio** |

When hosted on GitHub Pages or any standard static web server with HTTP gzip/brotli compression enabled, total network transfer is **under 500 KB**, ensuring instantaneous initial page load times.
