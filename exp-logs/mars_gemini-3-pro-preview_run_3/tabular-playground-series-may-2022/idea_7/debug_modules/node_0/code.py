import sys
import os
import shutil
import warnings
import pandas as pd
import torch
import numpy as np

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure library path is correct
sys.path.append(os.getcwd())

# Import library components
from library.config import Config
from library.data import prepare_data
from library.engine import run_training, generate_submission
from library.utils import set_seed


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    print("Setting up demonstration configuration...")

    # Set seed for reproducibility
    set_seed(42)

    # Define a dedicated working directory for this execution
    # This ensures we don't conflict with other runs or existing files
    demo_working_dir = os.path.join(Config.WORKING_DIR, "demo_execution")
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override Config parameters for a fast, verifiable run
    Config.WORKING_DIR = demo_working_dir
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128
    Config.DEBUG_SAMPLE_SIZE = 2000  # Small subset for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Update paths to point to the demo directory
    Config.MODEL_SAVE_PATH = os.path.join(demo_working_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_working_dir, "submission.csv")

    # Update cache paths to avoid reading full-dataset caches if they exist
    Config.CACHE_TRAIN_PATH = os.path.join(demo_working_dir, "train_processed.parquet")
    Config.CACHE_VAL_PATH = os.path.join(demo_working_dir, "val_processed.parquet")
    Config.CACHE_TEST_PATH = os.path.join(demo_working_dir, "test_processed.parquet")

    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"Running with {Config.EPOCHS} epochs and batch size {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation
    # -------------------------------------------------------------------------
    print("\nPreparing data (Debug Mode)...")

    # load_cached_data=False ensures we run the feature engineering pipeline
    # debug=True ensures we only process a small subset of data
    train_loader, val_loader, test_loader = prepare_data(
        load_cached_data=False, debug=True
    )

    # Verify DataLoaders
    print("Verifying DataLoaders...")
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Validation loader should not be empty."
    assert len(test_loader) > 0, "Test loader should not be empty."

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    assert "x_cont" in sample_batch, "Batch missing continuous features."
    assert "x_cat" in sample_batch, "Batch missing categorical features."
    assert "target" in sample_batch, "Batch missing target."
    assert (
        sample_batch["x_cont"].shape[1] == Config.NUM_CONT_FEATURES
    ), f"Incorrect continuous feature dim. Expected {Config.NUM_CONT_FEATURES}, got {sample_batch['x_cont'].shape[1]}"

    print("Data preparation verified.")

    # -------------------------------------------------------------------------
    # 3. Model Training
    # -------------------------------------------------------------------------
    print("\nStarting training loop...")

    # Run training using the engine
    # This handles model instantiation, training loop, validation, and saving
    model = run_training(train_loader, val_loader, test_loader)

    # Verify Model Artifacts
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"
    print("Training completed and model saved.")

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    print("\nGenerating submission...")

    # CRITICAL: generate_submission reads IDs from Config.TEST_DATA_PATH.
    # Since we are in debug mode, our test_loader only has DEBUG_SAMPLE_SIZE predictions.
    # We must create a corresponding subset of the test CSV so the IDs match the predictions.

    full_test_df = pd.read_csv(Config.TEST_DATA_PATH)
    debug_test_df = full_test_df.iloc[: Config.DEBUG_SAMPLE_SIZE].copy()

    debug_test_csv_path = os.path.join(demo_working_dir, "test.csv")
    debug_test_df.to_csv(debug_test_csv_path, index=False)

    # Temporarily override the test path in Config to point to our debug subset
    original_test_path = Config.TEST_DATA_PATH
    Config.TEST_DATA_PATH = debug_test_csv_path

    try:
        generate_submission(model, test_loader)
    finally:
        # Restore path just in case
        Config.TEST_DATA_PATH = original_test_path

    # -------------------------------------------------------------------------
    # 5. Final Verification
    # -------------------------------------------------------------------------
    print("\nVerifying submission output...")

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check dimensions
    expected_rows = Config.DEBUG_SAMPLE_SIZE
    assert (
        len(submission_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(submission_df)}"

    # Check columns
    assert list(submission_df.columns) == [
        "id",
        "target",
    ], f"Invalid columns: {submission_df.columns}"

    # Check value ranges
    preds = submission_df["target"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    print("Submission verified successfully.")
    print("Demonstration finished.")


if __name__ == "__main__":
    main()
