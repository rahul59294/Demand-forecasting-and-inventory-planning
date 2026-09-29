"""
Final Production Forecast Generation (Weeks 1 to 13 beyond 2011-11-28)
Milestone 6 - Part B

Point forecast model: MA4 (selected based on performance parity, simplicity, and operational transparency)
Quantile models: LightGBM Quantile Regression (q=0.10 and q=0.90) trained on all available historical data through 2011-11-28.
Output: data/processed/final_forecast_next13weeks.parquet
"""

import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path


def generate_final_forecasts():
    start_time = time.time()
    print("=" * 80)
    print("STARTING FINAL DEPLOYMENT FORECAST GENERATION (NEXT 13 WEEKS)")
    print("=" * 80)

    # 1. Load Data
    fu_path = Path("data/processed/forecast_universe.parquet")
    ret_sku_path = Path("data/processed/weekly_sku_demand_retail.parquet")
    cal_path = Path("data/processed/calendar_weeks.parquet")

    fu = pd.read_parquet(fu_path)
    ret_sku = pd.read_parquet(ret_sku_path)
    cal = pd.read_parquet(cal_path)

    print(f"Loaded Forecast Universe: {len(fu):,} SKUs")
    print(f"Loaded Calendar Weeks: {len(cal)} weeks ({cal['week_start'].min()} to {cal['week_start'].max()})")

    # Sort calendar
    cal = cal.sort_values("week_start").reset_index(drop=True)
    cal["week_idx"] = np.arange(len(cal))

    # Pre-encode categorical features
    abc_map = {"A": 0, "B": 1}
    pat_map = {"smooth": 0, "erratic": 1, "intermittent": 2, "lumpy": 3, "inactive": 4}
    
    fu["abc_code"] = fu["abc_class"].map(abc_map).fillna(1).astype(int)
    fu["pat_code"] = fu["demand_pattern"].map(pat_map).fillna(3).astype(int)
    fu["pat_deseas_code"] = fu["demand_pattern_deseasonalized"].map(pat_map).fillna(3).astype(int)
    sku_static_arr = fu[["abc_code", "pat_code", "pat_deseas_code", "median_price"]].values

    # Build 2D demand matrix: (N_skus, 104 weeks)
    date_to_idx = {w: i for i, w in enumerate(cal["week_start"])}
    skus_list = fu["StockCode"].tolist()
    sku_to_idx = {sku: i for i, sku in enumerate(skus_list)}
    num_skus = len(skus_list)
    num_weeks = len(cal)

    mat = np.zeros((num_skus, num_weeks), dtype=np.float32)
    for _, row in ret_sku.iterrows():
        s = row["StockCode"]
        w = row["week_start"]
        if s in sku_to_idx and w in date_to_idx:
            w_idx = date_to_idx[w]
            mat[sku_to_idx[s], w_idx] = float(row["retail_qty"])

    is_closed_arr = cal["is_closed_week"].values

    # Precompute active open weeks per SKU through week 103
    open_weeks_matrix = np.repeat(~is_closed_arr[np.newaxis, :], num_skus, axis=0)
    has_sold_matrix = np.cumsum(mat > 0, axis=1) > 0
    active_open_weeks_mat = np.cumsum(open_weeks_matrix & has_sold_matrix, axis=1)

    # Future 13 weeks calendar info (Weeks 104 to 116)
    last_week_start = pd.Timestamp("2011-11-28")
    future_weeks = [last_week_start + pd.Timedelta(weeks=h) for h in range(1, 14)]
    future_cal_df = pd.DataFrame({
        "week_ahead": np.arange(1, 14),
        "week_start": future_weeks,
        "month": [w.month for w in future_weeks],
        "iso_week": [w.isocalendar()[1] for w in future_weeks],
        "is_closed_week": [False] * 13 # Assume regular open weeks
    })
    
    # Weeks to Christmas calculation for future weeks
    future_wtc = []
    for w in future_weeks:
        c_year = w.year
        c_day = pd.Timestamp(year=c_year, month=12, day=25)
        diff_days = (c_day - w).days
        wtc = max(0, int(np.ceil(diff_days / 7.0)))
        future_wtc.append(wtc)
    future_cal_df["weeks_to_christmas"] = future_wtc

    # -------------------------------------------------------------
    # 2. Point Forecast: MA4 (Production Model)
    # -------------------------------------------------------------
    print("\n--- COMPUTING PRODUCTION POINT FORECASTS (MA4) ---")
    ma4_point_forecasts = {}

    # For each SKU, calculate MA4 on the last 4 open weeks prior to deployment
    open_mask = ~is_closed_arr
    open_indices = np.where(open_mask)[0]
    last_4_open_idx = open_indices[-4:]

    for i, sku in enumerate(skus_list):
        vals_4 = mat[i, last_4_open_idx]
        ma4_val = float(np.mean(vals_4))

        # 99th percentile capping on positive training weeks if >= 20 positive weeks
        pos_vals = mat[i, :104][mat[i, :104] > 0]
        if len(pos_vals) >= 20:
            cap_val = float(np.percentile(pos_vals, 99))
            ma4_val = min(ma4_val, cap_val)

        ma4_point_forecasts[sku] = max(0.0, ma4_val)

    # -------------------------------------------------------------
    # 3. Train Quantile LightGBM Models on All Data through Week 103
    # -------------------------------------------------------------
    print("\n--- TRAINING FULL QUANTILE MODELS (P10, P90) THROUGH 2011-11-28 ---")

    def extract_origin_features(t_orig):
        feats = []
        for lag in [1, 2, 3, 4, 8, 13, 26, 52]:
            idx = t_orig - (lag - 1)
            feats.append(mat[:, idx] if idx >= 0 else np.zeros(num_skus, dtype=np.float32))
        for rw in [4, 8, 13]:
            st = max(0, t_orig - rw + 1)
            window = mat[:, st : t_orig + 1]
            feats.append(np.mean(window, axis=1))
            feats.append(np.std(window, axis=1))
        feats.append(active_open_weeks_mat[:, t_orig])
        return np.column_stack(feats)

    def assemble_horizon_features(base_feats, orig_t, h, is_future=False):
        if is_future:
            w_row = future_cal_df[future_cal_df["week_ahead"] == h].iloc[0]
            m = w_row["month"]
            iw = w_row["iso_week"]
            wtc = w_row["weeks_to_christmas"]
            closed = 0
        else:
            tgt_idx = orig_t + h
            w_row = cal.iloc[tgt_idx]
            m = w_row["month"]
            iw = w_row["iso_week"]
            c_day = pd.Timestamp(year=w_row["year"], month=12, day=25)
            diff_days = (c_day - w_row["week_start"]).days
            wtc = max(0, int(np.ceil(diff_days / 7.0)))
            closed = int(w_row["is_closed_week"])

        n = len(base_feats)
        cal_arr = np.column_stack([
            np.full(n, m, dtype=np.float32),
            np.full(n, iw, dtype=np.float32),
            np.full(n, wtc, dtype=np.float32),
            np.full(n, closed, dtype=np.float32),
            np.full(n, h, dtype=np.float32),
            np.full(n, 0 if h <= 4 else (1 if h <= 8 else 2), dtype=np.float32)
        ])
        return np.hstack([base_feats, cal_arr, sku_static_arr])

    # Build full training set across multiple historical origins
    # Origins from t=52 to t=90 (predicting up to 13 weeks ahead)
    train_origins = [t for t in range(52, 91, 2)]
    X_train_list, y_train_list = [], []
    for orig_t in train_origins:
        base_t = extract_origin_features(orig_t)
        for h in range(1, 14):
            tgt_t = orig_t + h
            if tgt_t <= 103:
                X_train_list.append(assemble_horizon_features(base_t, orig_t, h, is_future=False))
                y_train_list.append(mat[:, tgt_t])

    X_train_full = np.vstack(X_train_list)
    y_train_full = np.concatenate(y_train_list)
    print(f"Full training dataset built: {X_train_full.shape}")

    full_ds = lgb.Dataset(X_train_full, label=y_train_full, free_raw_data=False)
    
    # Train Quantile models (q=0.10, q=0.90)
    best_params = {
        "num_leaves": 31,
        "learning_rate": 0.08,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "verbosity": -1,
        "n_jobs": 4
    }

    print("Training Quantile P10 Model...")
    q10_params = {"objective": "quantile", "alpha": 0.10, **best_params}
    bst_q10 = lgb.train(q10_params, full_ds, num_boost_round=90)

    print("Training Quantile P90 Model...")
    q90_params = {"objective": "quantile", "alpha": 0.90, **best_params}
    bst_q90 = lgb.train(q90_params, full_ds, num_boost_round=90)

    # -------------------------------------------------------------
    # 4. Generate Predictions for Next 13 Weeks (Origin = Week 103)
    # -------------------------------------------------------------
    print("\n--- GENERATING DEPLOYMENT PREDICTIONS (Origin: 2011-11-28) ---")
    base_deploy = extract_origin_features(103)

    records = []
    for h in range(1, 14):
        X_deploy_h = assemble_horizon_features(base_deploy, 103, h, is_future=True)
        pred_p10 = np.maximum(0, bst_q10.predict(X_deploy_h))
        pred_p90 = np.maximum(0, bst_q90.predict(X_deploy_h))
        w_start = future_weeks[h - 1]

        for i, sku in enumerate(skus_list):
            pt_fc = ma4_point_forecasts[sku]
            p10_val = float(pred_p10[i])
            p90_val = float(pred_p90[i])

            # Ensure logical consistency: p10 <= p90 and p90 >= pt_fc
            p10_clean = min(p10_val, pt_fc)
            p90_clean = max(p90_val, pt_fc)

            records.append({
                "StockCode": sku,
                "week_start": w_start,
                "week_ahead": h,
                "point_forecast": float(pt_fc),
                "p10": float(p10_clean),
                "p90": float(p90_clean)
            })

    final_fc_df = pd.DataFrame(records)
    out_path = Path("data/processed/final_forecast_next13weeks.parquet")
    final_fc_df.to_parquet(out_path, index=False)
    print(f"\nSuccessfully generated and saved: {out_path}")
    print(f"Total Rows: {len(final_fc_df):,} ({len(fu):,} SKUs x 13 weeks)")
    print(f"Date Range: {final_fc_df['week_start'].min()} to {final_fc_df['week_start'].max()}")

    # Print sample
    print("\nSample Forecasts (First 5 Rows):")
    print(final_fc_df.head().to_string(index=False))

    print(f"\nFinal forecast generation completed in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    generate_final_forecasts()
