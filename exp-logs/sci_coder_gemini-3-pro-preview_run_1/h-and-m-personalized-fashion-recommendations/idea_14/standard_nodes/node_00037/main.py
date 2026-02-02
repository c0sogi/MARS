import pandas as pd
import numpy as np
import os
import sys
import gc
import scipy.sparse as sp
from datetime import datetime

# Import provided libraries
from library.config import Config
from library.data_utils import load_processed_data
from library.igdc_model import IGDCRecommender
from library.metrics import apk


def set_seed(seed=42):
    np.random.seed(seed)
    import random

    random.seed(seed)


def predict_batch(model, user_ids, batch_size=5000):
    """
    Custom prediction loop for validation users.
    Mimics the logic in IGDCRecommender.predict but returns a DataFrame
    instead of writing to a file.
    """
    # Map users to internal indices
    user_idxs = model.mapper.transform(user_ids, "user")

    # Filter out unknown users (should not happen if map is global)
    valid_mask = user_idxs != -1
    user_ids = user_ids[valid_mask]
    user_idxs = user_idxs[valid_mask]

    predictions = []
    total = len(user_idxs)

    # Process in batches to manage memory
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_u_idxs = user_idxs[start:end]
        batch_cust_ids = user_ids[start:end]

        # --- Stratum 3: Trend (Fallback) ---
        # Base score is the trend vector broadcasted to the batch
        batch_scores = np.tile(model.R_trend, (len(batch_u_idxs), 1))

        # --- Stratum 2: Inventory-Gated CF (Discovery) ---
        if model.U_intent is not None:
            u_intent_batch = model.U_intent[batch_u_idxs]
            if u_intent_batch.nnz > 0:
                # Dot product: (Batch x Items)
                cf = u_intent_batch.dot(model.S_long)
                if sp.issparse(cf):
                    cf = cf.toarray()

                # Apply Inventory Mask
                cf = cf * model.M_active

                # Scale to [100, 1000]
                cf = (
                    cf * (Config.SCORE_CF_MAX - Config.SCORE_CF_MIN)
                    + Config.SCORE_CF_MIN
                )

                batch_scores += cf

        # --- Stratum 1: Habitual Repurchase (Priors) ---
        if model.U_habit is not None:
            u_habit_batch = model.U_habit[batch_u_idxs]
            if u_habit_batch.nnz > 0:
                habit = u_habit_batch.toarray()
                # Apply Offset (> 2000) only to non-zero entries
                mask = habit > 0
                habit[mask] += Config.SCORE_HABIT_OFFSET
                batch_scores += habit

        # --- Ranking ---
        k = 12
        # argpartition for efficient top-k selection
        top_k_part = np.argpartition(batch_scores, -k, axis=1)[:, -k:]

        # Sort the top-k
        rows = np.arange(len(batch_scores))[:, None]
        top_k_vals = batch_scores[rows, top_k_part]
        sorter = np.argsort(top_k_vals, axis=1)[:, ::-1]
        top_k_idxs = top_k_part[rows, sorter]

        # Map back to string article IDs
        flat_idxs = top_k_idxs.flatten()
        flat_ids = model.mapper.inverse_transform(flat_idxs, "item")
        batch_preds = flat_ids.reshape(len(batch_u_idxs), k)

        for cid, preds in zip(batch_cust_ids, batch_preds):
            # Format as 10-digit strings
            pred_str = " ".join([f"{int(p):010d}" for p in preds])
            predictions.append({"customer_id": cid, "prediction": pred_str})

    return pd.DataFrame(predictions)


def run_validation():
    print("=== Starting Validation Phase ===")

    # 1. Define Split
    # Load validation data to determine available dates dynamically
    print("Loading Validation Data to determine date range...")
    # Read only t_dat to determine range efficiently
    df_dates = pd.read_csv(Config.VAL_CSV, usecols=["t_dat"])
    df_dates["t_dat"] = pd.to_datetime(df_dates["t_dat"])

    max_date = df_dates["t_dat"].max()
    print(f"Max date in validation data: {max_date}")

    # Define Target Week (Last 7 days of available data)
    target_end_dt = max_date
    target_start_dt = max_date - pd.Timedelta(days=6)

    # Define Reference Date (Day before target week starts)
    VALIDATION_REF_DATE = (target_start_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    VALIDATION_TARGET_START = target_start_dt.strftime("%Y-%m-%d")
    VALIDATION_TARGET_END = target_end_dt.strftime("%Y-%m-%d")

    print(f"Dynamic Validation Split:")
    print(f"  Training End (Ref Date): {VALIDATION_REF_DATE}")
    print(f"  Validation Target: {VALIDATION_TARGET_START} to {VALIDATION_TARGET_END}")

    # Save original Config state
    original_ref_date = Config.REFERENCE_DATE
    original_sim_cache = Config.CACHE_SIMILARITY_MATRIX
    original_mask_cache = Config.CACHE_INVENTORY_MASK
    original_trend_cache = Config.CACHE_GLOBAL_TREND

    # Override Config for Validation
    Config.REFERENCE_DATE = VALIDATION_REF_DATE
    # Redirect caches to avoid overwriting production files
    Config.CACHE_SIMILARITY_MATRIX = original_sim_cache.replace(".npz", "_val.npz")
    Config.CACHE_INVENTORY_MASK = original_mask_cache.replace(".npy", "_val.npy")
    Config.CACHE_GLOBAL_TREND = original_trend_cache.replace(".npy", "_val.npy")

    print(f"Training validation model with Reference Date: {Config.REFERENCE_DATE}")

    # 2. Train Model on Shifted Data
    # load_cached_data=True allows loading the heavy 'transactions_processed.parquet' which is date-invariant
    # but the matrix caches will be missed (due to name change), forcing rebuild for the new date.
    model = IGDCRecommender()
    model.fit(load_cached_data=True)

    # 3. Prepare Ground Truth
    print("Preparing Validation Ground Truth...")
    # Load val.csv with string dtype for article_id to match predictions
    df_val = pd.read_csv(Config.VAL_CSV, dtype={"article_id": str})
    df_val["t_dat"] = pd.to_datetime(df_val["t_dat"])

    # Filter for target week
    mask = (df_val["t_dat"] >= VALIDATION_TARGET_START) & (
        df_val["t_dat"] <= VALIDATION_TARGET_END
    )
    df_truth = df_val[mask].copy()

    # Ensure Ground Truth IDs are zero-padded to match predictions
    df_truth["article_id"] = df_truth["article_id"].astype(str).str.zfill(10)

    # Group by customer to get list of purchased items
    truth_series = df_truth.groupby("customer_id")["article_id"].apply(list)
    target_customers = truth_series.index.values

    print(f"Validation Target Users: {len(target_customers)}")

    if len(target_customers) == 0:
        print("No validation users found in the target window.")
        return 0.0, None

    # 4. Predict
    print("Generating Validation Predictions...")
    preds_df = predict_batch(model, target_customers)

    # 5. Compute Metric
    print("Computing MAP@12...")
    actuals = truth_series.tolist()

    # Align predictions with truth
    preds_df = preds_df.set_index("customer_id").reindex(target_customers).fillna("")
    predicted_lists = preds_df["prediction"].astype(str).str.split().tolist()

    # Calculate scores per user for failure analysis
    user_scores = []
    for a, p in zip(actuals, predicted_lists):
        score = apk(a, p, k=12)
        user_scores.append(score)

    mean_map = np.mean(user_scores)
    print(f"Final Validation Metric: {mean_map:.10f}")

    # 6. Failure Analysis Data Extraction
    # Get user features from the model matrices
    user_idxs = model.mapper.transform(target_customers, "user")

    # Count non-zero entries in sparse matrices (History Length)
    habit_len = np.diff(model.U_habit.indptr)[user_idxs]
    # Intent Length (Items viewed in last 2 weeks)
    intent_len = np.diff(model.U_intent.indptr)[user_idxs]

    analysis_df = pd.DataFrame(
        {
            "customer_id": target_customers,
            "ap": user_scores,
            "habit_len": habit_len,
            "intent_len": intent_len,
        }
    )

    # Clean up validation model to free memory
    del model
    gc.collect()

    # Restore Config
    Config.REFERENCE_DATE = original_ref_date
    Config.CACHE_SIMILARITY_MATRIX = original_sim_cache
    Config.CACHE_INVENTORY_MASK = original_mask_cache
    Config.CACHE_GLOBAL_TREND = original_trend_cache

    return mean_map, analysis_df


def perform_failure_analysis(analysis_df):
    print("\n=== Failure Analysis ===")
    if analysis_df is None or len(analysis_df) == 0:
        print("No analysis data available.")
        return

    # Load customer metadata for additional features (Age)
    cust_df = pd.read_csv(Config.CUSTOMERS_CSV)
    analysis_df = analysis_df.merge(
        cust_df[["customer_id", "age"]], on="customer_id", how="left"
    )

    # Fill missing age with median
    analysis_df["age"] = analysis_df["age"].fillna(analysis_df["age"].median())

    # Compute Correlations
    cols = ["ap", "habit_len", "intent_len", "age"]
    corr = analysis_df[cols].corr()["ap"].sort_values()

    print("Correlation with Error (Lower AP = Higher Error):")
    print(corr)

    # Binning analysis for History Length
    # Users with more history should ideally have higher AP
    try:
        analysis_df["habit_bin"] = pd.qcut(
            analysis_df["habit_len"], 5, duplicates="drop"
        )
        print("\nMean AP by History Length Bin:")
        print(analysis_df.groupby("habit_bin", observed=True)["ap"].mean())
    except Exception as e:
        print(f"Could not perform binning analysis: {e}")


def run_submission():
    print("\n=== Starting Submission Phase ===")
    print(f"Training model with Reference Date: {Config.REFERENCE_DATE}")

    # Initialize and fit on full data
    # This will use the standard cache paths defined in Config
    model = IGDCRecommender()
    model.fit(load_cached_data=True)

    # Predict for test set and save to submission.csv
    model.predict()
    print("Submission generated.")


def main():
    set_seed(42)

    # Ensure processed data exists globally first
    # This creates the ID maps that we want to keep consistent across runs
    print("Initializing Data...")
    load_processed_data(load_cached_data=True)

    # Run Validation
    val_score, analysis_df = run_validation()

    # Run Failure Analysis
    perform_failure_analysis(analysis_df)

    # Check Threshold and Submit
    THRESHOLD = 0.0265060791
    if val_score > THRESHOLD:
        print(
            f"\nValidation score {val_score:.10f} > {THRESHOLD}. Proceeding to submission."
        )
        run_submission()
    else:
        print(
            f"\nValidation score {val_score:.10f} <= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
