import os
import sys
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.data_loader import ForestDataLoader
from library.model import GBDTWrapper
from library.trainer import run_training


def main():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set random seed for reproducibility
    np.random.seed(42)

    print("=== Starting Library Usage Demonstration ===\n")

    # Ensure directories are set up
    print("1. Setting up directories...")
    Config.setup_directories()

    # Verify directories exist
    if not os.path.exists(Config.WORKING_DIR):
        raise FileNotFoundError(
            f"Working directory {Config.WORKING_DIR} was not created."
        )
    if not os.path.exists(Config.SUBMISSION_DIR):
        raise FileNotFoundError(
            f"Submission directory {Config.SUBMISSION_DIR} was not created."
        )
    print("   Directories verified.")

    # -------------------------------------------------------------------------
    # 2. Demonstrate Data Loader (with Debug Mode)
    # -------------------------------------------------------------------------
    print("\n2. Testing ForestDataLoader...")

    # We enable DEBUG mode to load only a small subset (10,000 rows) for speed
    Config.DEBUG = True
    print(f"   Debug mode enabled. Sample size: {Config.DEBUG_SAMPLE_SIZE}")

    loader = ForestDataLoader()

    # Load Training Data
    # load_cached_data=False forces the loader to read from raw parquet and process it
    print("   Loading training data...")
    X_train, y_train = loader.get_data("train", load_cached_data=False)

    # Validate Training Data
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(y_train, np.ndarray), "y_train should be a numpy array"
    assert (
        len(X_train) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows in train"
    assert len(y_train) == Config.DEBUG_SAMPLE_SIZE, "Mismatch in X and y lengths"
    # Check if target was correctly transformed (0-indexed)
    assert y_train.min() >= 0, "Target values should be 0-indexed (min >= 0)"

    # Load Test Data
    print("   Loading test data...")
    X_test, y_test = loader.get_data("test", load_cached_data=False)

    # Validate Test Data
    assert (
        len(X_test) == Config.DEBUG_SAMPLE_SIZE
    ), "Test set size mismatch in debug mode"
    assert y_test is None, "Test target should be None"

    print("   Data Loader assertions passed.")

    # -------------------------------------------------------------------------
    # 3. Demonstrate Model Wrapper
    # -------------------------------------------------------------------------
    print("\n3. Testing GBDTWrapper (XGBoost)...")

    # Initialize model with minimal hyperparameters for instant training
    # We override n_estimators to 2 and set verbosity to 0 (silent)
    model = GBDTWrapper(n_estimators=2, max_depth=3, learning_rate=0.1, verbosity=0)

    # Train the model
    print("   Training model (fast mode)...")
    model.fit(X_train, y_train)

    # Generate predictions on the training set just to verify the predict method
    preds = model.predict(X_train)

    # Validate Predictions
    assert len(preds) == len(X_train), "Prediction length mismatch"
    assert np.issubdtype(preds.dtype, np.integer), "Predictions should be integers"
    print("   Model training and prediction successful.")

    # -------------------------------------------------------------------------
    # 4. Demonstrate Full Pipeline (Trainer)
    # -------------------------------------------------------------------------
    print("\n4. Running Full Pipeline via run_training()...")

    # run_training encapsulates the entire flow: Load -> Train -> Predict -> Submit
    # We pass debug=True to ensure it uses the small subset.
    # We pass load_cached_data=False to ensure it runs the logic freshly.
    run_training(
        debug=True,
        load_cached_data=False,
        n_estimators=5,  # Small number for speed
        learning_rate=0.05,
        max_depth=3,
        verbosity=0,  # Silent mode
    )
    print("   Pipeline execution complete.")

    # -------------------------------------------------------------------------
    # 5. Validate Submission File
    # -------------------------------------------------------------------------
    print("\n5. Validating Submission File...")

    submission_path = Config.SUBMISSION_FILE_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    # Check shape
    print(f"   Submission shape: {df_sub.shape}")
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission should have {Config.DEBUG_SAMPLE_SIZE} rows in debug mode"

    # Check columns
    assert Config.ID_COL in df_sub.columns, f"Missing ID column: {Config.ID_COL}"
    assert (
        Config.TARGET_COL in df_sub.columns
    ), f"Missing Target column: {Config.TARGET_COL}"

    # Check ID correctness (should match the first N IDs of test.parquet)
    # We read the raw test file to compare
    df_test_raw = pd.read_parquet(Config.TEST_DATA_PATH)
    expected_ids = df_test_raw[Config.ID_COL].iloc[: Config.DEBUG_SAMPLE_SIZE].values
    actual_ids = df_sub[Config.ID_COL].values
    np.testing.assert_array_equal(
        actual_ids, expected_ids, err_msg="Submission IDs do not match Test IDs"
    )

    # Check Target values (Should be 1-based, i.e., >= 1)
    # The model predicts 0-6, but generate_submission maps it to 1-7
    min_val = df_sub[Config.TARGET_COL].min()
    max_val = df_sub[Config.TARGET_COL].max()
    print(f"   Predicted classes range: {min_val} to {max_val}")

    assert (
        min_val >= 1
    ), "Submission contains class 0 or negative, expected 1-based indexing."

    print("   Submission file is valid.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
