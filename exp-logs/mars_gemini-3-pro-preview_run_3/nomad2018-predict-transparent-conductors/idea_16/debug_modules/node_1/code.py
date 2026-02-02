import os
import sys
import shutil
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch

# Import library modules
import library.config as config
import library.model as model_lib
import library.data_loader as loader_lib

# =============================================================================
# Configuration & Setup
# =============================================================================

# Create a temporary directory for this run to avoid affecting existing caches
TEMP_WORKING_DIR = tempfile.mkdtemp(prefix="demo_working_")
print(f"Using temporary working directory: {TEMP_WORKING_DIR}")

# Override global configuration to use the temp directory
config.WORKING_DIR = TEMP_WORKING_DIR
config.TRAIN_FEATURES_PATH = os.path.join(TEMP_WORKING_DIR, "train_features.parquet")
config.VAL_FEATURES_PATH = os.path.join(TEMP_WORKING_DIR, "val_features.parquet")
config.TEST_FEATURES_PATH = os.path.join(TEMP_WORKING_DIR, "test_features.parquet")

# Override XGBoost parameters for extremely fast training
FAST_XGB_PARAMS = {
    "n_estimators": 5,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "objective": "reg:squarederror",
    "n_jobs": 1,
    "random_state": 42,
    "tree_method": "hist",
    "verbosity": 0,
}
config.XGB_PARAMS = FAST_XGB_PARAMS

# =============================================================================
# Mocking Data Loading
# =============================================================================

# Save reference to the original read_csv
original_read_csv = pd.read_csv


def mocked_read_csv(filepath_or_buffer, *args, **kwargs):
    """
    Intercepts pandas.read_csv calls.
    If a metadata file is requested, returns only the first 10 rows.
    This drastically reduces the time spent on feature extraction (XYZ processing).
    """
    df = original_read_csv(filepath_or_buffer, *args, **kwargs)

    s_path = str(filepath_or_buffer)
    if (
        s_path.endswith("train_metadata.csv")
        or s_path.endswith("val_metadata.csv")
        or s_path.endswith("test_metadata.csv")
    ):
        print(f"  [Mock] Truncating {os.path.basename(s_path)} to 10 rows for speed.")
        return df.head(10)

    return df


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("Starting pipeline demonstration...")

    # Apply the patch to pandas.read_csv
    with patch("pandas.read_csv", side_effect=mocked_read_csv):

        # ---------------------------------------------------------
        # 1. Train Model (includes Data Loading & Feature Extraction)
        # ---------------------------------------------------------
        print("\n--- Step 1: Training Model ---")
        # We disable loading from cache to demonstrate the feature extraction logic
        # on our small 10-sample datasets.
        model = model_lib.train_model(load_cached_data=False)

        # Verify model structure
        assert isinstance(model, model_lib.DualTargetRegressor)
        for target in config.TARGET_COLS:
            assert target in model.models
            print(f"  Regressor for '{target}' trained.")

        # ---------------------------------------------------------
        # 2. Manual Prediction Check
        # ---------------------------------------------------------
        print("\n--- Step 2: Verifying Predictions ---")
        # Load the (mocked/truncated) validation set
        X_val, y_val = loader_lib.load_and_process_data("val", load_cached_data=True)

        # Generate predictions
        preds = model.predict(X_val)

        print(f"  Input shape: {X_val.shape}")
        print(f"  Output shape: {preds.shape}")
        print("  Sample predictions:")
        print(preds.head())

        # Assertions to ensure logic correctness
        assert len(preds) == 10, "Prediction count mismatch (expected 10 from mock)."
        assert list(preds.columns) == config.TARGET_COLS, "Incorrect target columns."
        assert not preds.isnull().values.any(), "Predictions contain NaNs."

        # ---------------------------------------------------------
        # 3. Generate Submission
        # ---------------------------------------------------------
        print("\n--- Step 3: Generating Submission ---")
        # This processes the test set (truncated to 10 rows) and saves CSV
        model_lib.generate_submission(model, load_cached_data=False)

        # Verify submission file
        if os.path.exists(config.SUBMISSION_PATH):
            sub_df = pd.read_csv(config.SUBMISSION_PATH)
            print(f"  Submission file created at: {config.SUBMISSION_PATH}")
            print(f"  Submission shape: {sub_df.shape}")

            expected_cols = ["id"] + config.TARGET_COLS
            assert all(
                col in sub_df.columns for col in expected_cols
            ), "Missing columns in submission."
            assert len(sub_df) == 10, "Submission should have 10 rows (from mock)."
        else:
            raise FileNotFoundError("Submission file was not created.")

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    print("\n--- Cleanup ---")
    try:
        shutil.rmtree(TEMP_WORKING_DIR)
        print(f"Removed temporary directory: {TEMP_WORKING_DIR}")
    except Exception as e:
        print(f"Warning: Failed to remove temp dir: {e}")

    print("\nDemonstration completed successfully.")
