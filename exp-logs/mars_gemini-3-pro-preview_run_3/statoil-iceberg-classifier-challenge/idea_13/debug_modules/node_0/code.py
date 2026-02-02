import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_data_loaders
from library.model import DSN_CNN
from library.trainer import run_kfold_training

if __name__ == "__main__":
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 1] Overriding Configuration for Speed...")

    # Set to Debug mode to use a small subset of data (100 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 100

    # Reduce training complexity
    Config.N_FOLDS = 2  # Only run 2 folds instead of 5
    Config.NUM_EPOCHS = 2  # Only run 2 epochs per fold
    Config.BATCH_SIZE = 16  # Smaller batch size
    Config.PATIENCE = 2  # Short patience

    # Use a specific directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working"
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Update cache paths to point to the demo directory so we don't overwrite main cache
    # or rely on potentially missing cache in a new directory
    Config.CACHE_TRAIN_X = os.path.join(Config.WORKING_DIR, "X_train.npy")
    Config.CACHE_TRAIN_Y = os.path.join(Config.WORKING_DIR, "y_train.npy")
    Config.CACHE_TRAIN_ANGLE = os.path.join(Config.WORKING_DIR, "angles_train.npy")
    Config.CACHE_VAL_X = os.path.join(Config.WORKING_DIR, "X_val.npy")
    Config.CACHE_VAL_Y = os.path.join(Config.WORKING_DIR, "y_val.npy")
    Config.CACHE_VAL_ANGLE = os.path.join(Config.WORKING_DIR, "angles_val.npy")
    Config.CACHE_TEST_X = os.path.join(Config.WORKING_DIR, "X_test.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.WORKING_DIR, "ids_test.npy")
    Config.CACHE_TEST_ANGLE = os.path.join(Config.WORKING_DIR, "angles_test.npy")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated for demo run.")

    # -------------------------------------------------------------------------
    # 2. Component Verification: Model
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Model Architecture...")

    # Instantiate model
    model = DSN_CNN()
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input: Batch size 4, 3 channels (HH, HV, Avg), 75x75 image
    dummy_img = torch.randn(4, 3, 75, 75).to(Config.DEVICE)
    # Dummy angles: Batch size 4, 1 value
    dummy_angle = torch.randn(4, 1).to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_img, dummy_angle)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    if output.shape != (4, 1):
        raise AssertionError(f"Expected output shape (4, 1), got {output.shape}")

    print("Model verification successful.")

    # -------------------------------------------------------------------------
    # 3. Component Verification: Data Loader
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Data Loading...")

    # We use get_data_loaders which handles loading, processing, and batching
    # Since we set Config.DEBUG = True, this will load the truncated dataset
    train_loader, val_loader, test_loader, ids_test = get_data_loaders(
        load_cached_data=False, debug=True
    )

    # Fetch one batch
    images, angles, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Angle Shape: {angles.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Assertions
    if images.shape[1:] != (3, 75, 75):
        raise AssertionError(
            f"Expected image dimensions (C, H, W) = (3, 75, 75), got {images.shape[1:]}"
        )
    if angles.shape[1] != 1:
        raise AssertionError("Expected angle dimension to be 1")

    print("Data Loader verification successful.")

    # -------------------------------------------------------------------------
    # 4. Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Full K-Fold Training Pipeline (Demo Mode)...")

    # This function orchestrates the training, validation, and prediction
    # It uses the Config settings we modified earlier
    run_kfold_training()

    print("Pipeline execution completed.")

    # -------------------------------------------------------------------------
    # 5. Output Verification
    # -------------------------------------------------------------------------
    print("\n[Step 5] Verifying Submission Output...")

    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_FILE}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print("First 5 rows:")
    print(df_sub.head())

    # Check columns
    expected_cols = ["id", "is_iceberg"]
    if list(df_sub.columns) != expected_cols:
        raise AssertionError(
            f"Expected columns {expected_cols}, got {list(df_sub.columns)}"
        )

    # Check values
    if df_sub["is_iceberg"].min() < 0 or df_sub["is_iceberg"].max() > 1:
        raise AssertionError("Probabilities in 'is_iceberg' must be between 0 and 1")

    # Check length (Should match DEBUG_SAMPLES because we truncated test data too in Config.DEBUG mode)
    if len(df_sub) != Config.DEBUG_SAMPLES:
        raise AssertionError(
            f"Expected {Config.DEBUG_SAMPLES} rows in submission (Debug Mode), got {len(df_sub)}"
        )

    print("\n=== Demonstration Completed Successfully ===")
