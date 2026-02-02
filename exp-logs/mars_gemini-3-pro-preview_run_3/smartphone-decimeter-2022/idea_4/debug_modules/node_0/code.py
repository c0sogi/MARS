import os
import sys
import pandas as pd
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.data_loader import load_metadata
from library.feature_engineering import generate_dataset
from library.model import ResidualRegressor
from library.kalman_filter import apply_kalman_smoothing
from library.evaluation import calculate_metric


def setup_demo_environment():
    """
    Configures the environment for a fast demonstration run.
    Modifies the Config class attributes directly.
    """
    print("--- Setting up Demo Environment ---")

    # Set random seed for reproducibility
    Config.SEED = 42
    np.random.seed(Config.SEED)

    # Define a specific working directory for this demo to avoid overwriting existing work
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update cache paths to point to the demo directory
    Config.TRAIN_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.TRAIN_TARGETS_PATH = os.path.join(
        Config.WORKING_DIR, "train_targets.parquet"
    )
    Config.VAL_FEATURES_PATH = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.VAL_TARGETS_PATH = os.path.join(Config.WORKING_DIR, "val_targets.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Optimization for speed: Use a very small subset of trips
    # This limits the data processing to only 3 unique trips per split
    Config.DEBUG_SAMPLE_SIZE = 3

    # Optimization for speed: Reduce LightGBM complexity
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["min_child_samples"] = 5
    Config.LGBM_PARAMS["num_leaves"] = 16

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Model Estimators: {Config.LGBM_PARAMS['n_estimators']}")


def run_demo_pipeline():
    # 1. Load Metadata
    print("\n--- 1. Loading Metadata ---")
    try:
        train_meta = load_metadata("train")
        val_meta = load_metadata("val")
        test_meta = load_metadata("test")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the metadata generation script has been run.")
        return

    print(f"Train Metadata Rows: {len(train_meta)}")
    print(f"Val Metadata Rows: {len(val_meta)}")
    print(f"Test Metadata Rows: {len(test_meta)}")

    # 2. Feature Engineering (Train & Val)
    print("\n--- 2. Generating Features (Train/Val) ---")
    # load_cached_data=False forces regeneration for this demo to ensure logic runs
    X_train, y_train = generate_dataset(
        train_meta, mode="train", load_cached_data=False
    )
    X_val, y_val = generate_dataset(val_meta, mode="val", load_cached_data=False)

    print(f"Train Features Shape: {X_train.shape}")
    print(f"Val Features Shape: {X_val.shape}")

    if len(X_train) == 0 or len(X_val) == 0:
        raise RuntimeError(
            "Dataset generation failed (empty dataframe). Check input data."
        )

    # 3. Model Training
    print("\n--- 3. Training Model ---")
    model = ResidualRegressor()
    model.fit(X_train, y_train, X_val, y_val)

    # 4. Validation Prediction & Smoothing
    print("\n--- 4. Predicting and Smoothing (Validation) ---")
    # Predict residuals
    preds_res = model.predict(X_val)

    # Construct absolute predictions (WLS Baseline + Residual)
    val_preds = X_val[["tripId", "UnixTimeMillis", "wls_lat", "wls_lon"]].copy()
    val_preds["LatitudeDegrees"] = val_preds["wls_lat"] + preds_res["pred_lat_res"]
    val_preds["LongitudeDegrees"] = val_preds["wls_lon"] + preds_res["pred_lon_res"]

    # Apply Kalman Smoothing
    val_preds_smoothed = apply_kalman_smoothing(val_preds)

    # Verify smoothing didn't change shape
    assert val_preds.shape == val_preds_smoothed.shape

    # 5. Evaluation
    print("\n--- 5. Evaluation ---")
    # Reconstruct Ground Truth for Validation set from features and targets
    # y_val contains (GT - WLS), so GT = WLS + y_val
    val_gt = X_val[["tripId", "UnixTimeMillis"]].copy()
    val_gt["LatitudeDegrees"] = X_val["wls_lat"] + y_val["target_lat"]
    val_gt["LongitudeDegrees"] = X_val["wls_lon"] + y_val["target_lon"]

    # Calculate Scores
    raw_score = calculate_metric(val_preds, val_gt)
    smooth_score = calculate_metric(val_preds_smoothed, val_gt)

    print(f"Raw Validation Score:      {raw_score:.4f}")
    print(f"Smoothed Validation Score: {smooth_score:.4f}")

    # Assertions to ensure logic holds
    assert not np.isnan(raw_score), "Raw score is NaN"
    assert not np.isnan(smooth_score), "Smoothed score is NaN"

    # 6. Test Inference (Demonstration)
    print("\n--- 6. Test Inference ---")
    # Generate test features
    X_test, _ = generate_dataset(test_meta, mode="test", load_cached_data=False)

    if not X_test.empty:
        # Predict
        test_res = model.predict(X_test)

        # Construct final submission dataframe
        submission = X_test[["tripId", "UnixTimeMillis", "wls_lat", "wls_lon"]].copy()
        submission["LatitudeDegrees"] = submission["wls_lat"] + test_res["pred_lat_res"]
        submission["LongitudeDegrees"] = (
            submission["wls_lon"] + test_res["pred_lon_res"]
        )

        # Apply Smoothing
        submission = apply_kalman_smoothing(submission)

        # Select required columns
        submission = submission[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ]

        # Save
        submission.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Demo submission saved to: {Config.SUBMISSION_FILE}")
        print(f"Submission Shape: {submission.shape}")
    else:
        print(
            "Test dataset empty (likely due to sampling), skipping submission generation."
        )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    setup_demo_environment()
    run_demo_pipeline()
