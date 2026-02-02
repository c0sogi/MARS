import os
import sys
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.feature_engineering import extract_features_for_segment
from library.data_loader import process_dataset
from library.model import run_cross_validation, predict_ensemble


def run_demo():
    # -------------------------------------------------------------------------
    # 0. Setup and Configuration Override
    # -------------------------------------------------------------------------
    print("Initializing Demo Configuration...")

    # Set fixed seeds
    np.random.seed(42)

    # Override Config for speed and small data compatibility
    # We use a specific subdirectory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Reduce model complexity for the tiny demo dataset (20 samples)
    # Standard LightGBM params (like num_leaves=128) would fail or warn on 20 rows
    Config.MODEL_PARAMS["n_estimators"] = 10
    Config.MODEL_PARAMS["early_stopping_rounds"] = 5
    Config.MODEL_PARAMS["num_leaves"] = 4
    Config.MODEL_PARAMS["min_child_samples"] = 2
    Config.MODEL_PARAMS["learning_rate"] = 0.05
    Config.N_FOLDS = 2  # Use 2 folds for quick validation

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model Params: {Config.MODEL_PARAMS}")

    # -------------------------------------------------------------------------
    # 1. Demonstrate Feature Engineering (Unit Level)
    # -------------------------------------------------------------------------
    print("\n--- Step 1: Feature Engineering (Single Segment) ---")

    # Load one raw file manually to test the extraction logic
    train_meta = pd.read_csv(Config.TRAIN_META_PATH)
    sample_row = train_meta.iloc[0]
    sample_file_path = os.path.join(Config.INPUT_DIR, sample_row["file_path"])

    print(f"Loading raw data from: {sample_file_path}")
    raw_df = pd.read_csv(sample_file_path, dtype="float32")

    # Extract features
    features = extract_features_for_segment(raw_df)

    # Validation
    assert isinstance(features, dict), "Feature extraction should return a dictionary"

    # Check for specific feature keys defined in feature_engineering.py
    # e.g., sensor_1_mean is not directly there, but sensor_1_min, sensor_1_trend_q50 are
    expected_keys = ["sensor_1_min", "sensor_1_trend_q50", "sensor_10_resid_rms"]
    for key in expected_keys:
        assert key in features, f"Missing expected feature: {key}"

    print(
        f"Successfully extracted {len(features)} features from segment {sample_row['segment_id']}"
    )

    # -------------------------------------------------------------------------
    # 2. Demonstrate Data Loading Pipeline (Integration Level)
    # -------------------------------------------------------------------------
    print("\n--- Step 2: Data Loading Pipeline (Batch Processing) ---")

    # Process a small subset (debug_size=20) of the training data
    # This tests parallel processing and caching
    train_df = process_dataset(
        Config.TRAIN_META_PATH,
        load_cached_data=False,  # Force processing
        is_test=False,
        debug_size=20,
        n_jobs=2,  # Use 2 cores for demo
    )

    # Validation
    assert not train_df.empty, "Training DataFrame should not be empty"
    assert len(train_df) == 20, f"Expected 20 rows, got {len(train_df)}"
    assert (
        "time_to_eruption" in train_df.columns
    ), "Target column missing in training data"
    assert "segment_id" in train_df.columns, "Segment ID missing"

    # Check that features are numeric and not all NaN
    feature_cols = [
        c for c in train_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    assert len(feature_cols) > 0, "No feature columns found"
    assert not train_df[feature_cols].isnull().all().all(), "Features contain only NaNs"

    print(f"Processed Training Data Shape: {train_df.shape}")

    # -------------------------------------------------------------------------
    # 3. Demonstrate Model Training (Cross-Validation)
    # -------------------------------------------------------------------------
    print("\n--- Step 3: Model Training (Cross-Validation) ---")

    # Run CV with the reduced dataset and reduced parameters
    models = run_cross_validation(train_df)

    # Validation
    assert isinstance(models, list), "run_cross_validation should return a list"
    assert (
        len(models) == Config.N_FOLDS
    ), f"Expected {Config.N_FOLDS} models, got {len(models)}"

    print("Cross-validation completed successfully.")

    # -------------------------------------------------------------------------
    # 4. Demonstrate Inference
    # -------------------------------------------------------------------------
    print("\n--- Step 4: Inference on Test Data ---")

    # Process a small subset of test data
    test_df = process_dataset(
        Config.TEST_META_PATH,
        load_cached_data=False,
        is_test=True,
        debug_size=5,
        n_jobs=2,
    )

    # Validation
    assert (
        "time_to_eruption" not in test_df.columns
    ), "Target column should not be in test data"
    assert len(test_df) == 5, "Expected 5 test samples"

    # Generate predictions
    # We need to ensure the columns match exactly what the model expects
    # The model was trained on feature_cols from train_df
    # process_dataset ensures consistent feature extraction, so columns should match
    # (excluding target and segment_id)
    feature_cols_test = [c for c in test_df.columns if c != "segment_id"]

    # Verify column alignment (simple check)
    train_feats = set(train_df.columns) - {"segment_id", "time_to_eruption"}
    test_feats = set(test_df.columns) - {"segment_id"}
    assert train_feats == test_feats, "Feature mismatch between train and test sets"

    # Predict
    X_test = test_df[
        list(train_feats)
    ]  # Ensure order doesn't matter by selecting explicitly
    preds = predict_ensemble(models, X_test)

    # Validation
    assert len(preds) == 5, "Prediction count mismatch"
    assert np.issubdtype(preds.dtype, np.number), "Predictions should be numeric"
    assert not np.isnan(preds).any(), "Predictions contain NaNs"

    # Create submission dataframe
    submission = pd.DataFrame(
        {"segment_id": test_df["segment_id"], "time_to_eruption": preds}
    )

    print("Sample Predictions:")
    print(submission)

    # Save to working directory
    sub_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Demo submission saved to {sub_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
