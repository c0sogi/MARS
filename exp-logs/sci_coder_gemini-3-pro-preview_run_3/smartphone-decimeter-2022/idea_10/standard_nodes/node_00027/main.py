import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import warnings

# Import from provided libraries
from library.config import SEED, WORKING_DIR
from library.utils import setup_logger, haversine_distance
from library.feature_engineering import prepare_dataset
from library.model import train_model, predict_residuals, get_feature_columns
from library.optimizer import optimize_trajectory, save_submission

# Suppress warnings
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def calculate_metric(df):
    """
    Calculates the competition metric: mean of the 50th and 95th percentile distance errors.
    """
    # Calculate distance error for each point
    # We assume 'LatitudeDegrees' and 'LongitudeDegrees' are the PREDICTED values
    # and we need to compare against Ground Truth.

    if "gt_lat" not in df.columns or "gt_lon" not in df.columns:
        raise ValueError(
            "Ground truth columns 'gt_lat', 'gt_lon' missing for metric calculation."
        )

    df["dist_error"] = haversine_distance(
        df["LatitudeDegrees"], df["LongitudeDegrees"], df["gt_lat"], df["gt_lon"]
    )

    # Group by tripId (phone)
    trip_metrics = []
    for trip_id, group in df.groupby("tripId"):
        errors = group["dist_error"].values
        p50 = np.percentile(errors, 50)
        p95 = np.percentile(errors, 95)
        trip_metrics.append((p50 + p95) / 2)

    return np.mean(trip_metrics)


def run_failure_analysis(df, feature_cols):
    """
    Correlates error magnitude with features.
    """
    print("\n--- Failure Analysis ---")
    # Calculate distance error if not already present
    if "dist_error" not in df.columns:
        if "gt_lat" in df.columns:
            df["dist_error"] = haversine_distance(
                df["LatitudeDegrees"],
                df["LongitudeDegrees"],
                df["gt_lat"],
                df["gt_lon"],
            )
        else:
            print("Cannot run failure analysis: Ground truth missing.")
            return

    # Select numerical features for correlation
    # Filter only columns that exist in df and are in feature_cols
    valid_cols = [c for c in feature_cols if c in df.columns]

    if not valid_cols:
        print("No feature columns found for analysis.")
        return

    analysis_df = df[valid_cols + ["dist_error"]].dropna()

    if analysis_df.empty:
        print("No valid data for failure analysis (empty after dropna).")
        return

    correlations = analysis_df.corr()["dist_error"].sort_values(ascending=False)

    print("Top 10 features positively correlated with Error Magnitude:")
    print(correlations.head(11).iloc[1:])  # Skip self-correlation
    print("\nTop 5 negatively correlated features:")
    print(correlations.tail(5))


def main():
    # Setup
    logger = setup_logger()
    set_seed(SEED)
    logger.info("Starting runfile.py execution...")

    # FIX: Remove stale dataset caches to ensure tripId is added
    # Cite debug_lesson_3: Invalidate Data Caches When Modifying Data Processing Logic
    for split in ["train", "val", "test"]:
        cache_path = os.path.join(WORKING_DIR, f"{split}_dataset.parquet")
        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info(f"Deleted stale cache: {cache_path}")

    # 1. Prepare Data
    logger.info("Loading and preparing training data...")
    # load_cached_data=True will look for parquet files in working/idea_10/
    X_train, y_train = prepare_dataset("train", load_cached_data=True)

    logger.info("Loading and preparing validation data...")
    X_val, y_val = prepare_dataset("val", load_cached_data=True)

    # 2. Train Model
    # We use drive_id as groups for GroupKFold
    if "drive_id" not in X_train.columns:
        logger.error("drive_id column missing in training data!")
        return

    groups = X_train["drive_id"]

    logger.info("Training LightGBM models...")
    models = train_model(X_train, y_train, groups)

    # 3. Validation & Optimization
    logger.info("Predicting residuals on validation set...")
    val_preds = predict_residuals(models, X_val)

    # Add predictions to validation dataframe
    X_val["pred_e"] = val_preds["pred_e"]
    X_val["pred_n"] = val_preds["pred_n"]

    # Store Ground Truth before optimization overwrites Lat/Lon
    # The prepare_dataset function returns X_val with 'LatitudeDegrees' as GT
    X_val["gt_lat"] = X_val["LatitudeDegrees"].copy()
    X_val["gt_lon"] = X_val["LongitudeDegrees"].copy()

    logger.info("Optimizing validation trajectories (Global L1)...")
    # This updates LatitudeDegrees/LongitudeDegrees in X_val with optimized values
    val_optimized = optimize_trajectory(X_val)

    # 4. Metric Calculation
    score = calculate_metric(val_optimized)
    print(f"Final Validation Metric: {score}")

    # 5. Failure Analysis
    feature_cols = get_feature_columns(X_train)
    run_failure_analysis(val_optimized, feature_cols)

    # 6. Submission
    THRESHOLD = 4.32379283550646
    if score < THRESHOLD:
        logger.info(f"Validation score {score} < {THRESHOLD}. Generating submission...")

        logger.info("Loading and preparing test data...")
        X_test, _ = prepare_dataset("test", load_cached_data=True)

        logger.info("Predicting test residuals...")
        test_preds = predict_residuals(models, X_test)

        X_test["pred_e"] = test_preds["pred_e"]
        X_test["pred_n"] = test_preds["pred_n"]

        logger.info("Optimizing test trajectories...")
        test_optimized = optimize_trajectory(X_test)

        save_submission(test_optimized, "./submission/submission.csv")
    else:
        logger.info(
            f"Validation score {score} >= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()
