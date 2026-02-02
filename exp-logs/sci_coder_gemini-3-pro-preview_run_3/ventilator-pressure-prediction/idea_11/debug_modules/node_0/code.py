import os
import shutil
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library files
from library.config import Config
from library.dataset import get_data_loaders
from library.model import PMNCNet
from library.trainer import Trainer, generate_submission


def main():
    print("Starting Ventilator Pressure Prediction Demo...")

    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Isolation
    # --------------------------------------------------------------------------
    # We modify the Config class at runtime to create a fast, isolated demo environment.

    # Enable Debug mode to use a small subset of data (1000 breaths)
    Config.DEBUG = True

    # Reduce training duration for demonstration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16  # Smaller batch size for the small debug dataset

    # Define a specific working directory for this demo to avoid file conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Update Config paths to point to the demo directory
    # Note: We must update these explicitly because they were initialized at import time
    Config.WORKING_DIR = demo_dir
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.LAST_MODEL_PATH = os.path.join(demo_dir, "last_model.pth")
    Config.CACHE_TRAIN_X = os.path.join(demo_dir, "train_x.npy")
    Config.CACHE_TRAIN_Y = os.path.join(demo_dir, "train_y.npy")
    Config.CACHE_VAL_X = os.path.join(demo_dir, "val_x.npy")
    Config.CACHE_VAL_Y = os.path.join(demo_dir, "val_y.npy")
    Config.CACHE_TEST_X = os.path.join(demo_dir, "test_x.npy")
    Config.CACHE_TEST_IDS = os.path.join(demo_dir, "test_ids.npy")
    Config.SCALER_PATH = os.path.join(demo_dir, "scaler_stats.npz")
    Config.SUBMISSION_FILE = os.path.join(demo_dir, "submission.csv")

    print(f"Working directory set to: {Config.WORKING_DIR}")

    # --------------------------------------------------------------------------
    # 2. Data Loading and Processing
    # --------------------------------------------------------------------------
    print("\n[Step 1] Initializing DataLoaders...")
    # load_cached_data=False forces the pipeline to run: CSV Load -> Feature Eng -> Scaling -> Reshape
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        load_cached_data=False
    )

    # Verify Data Shapes
    print("Verifying data shapes...")
    try:
        x_batch, y_batch = next(iter(train_loader))

        # Expected Input: (Batch, Seq_Len=80, Features)
        assert x_batch.dim() == 3, f"Input must be 3D, got {x_batch.shape}"
        assert (
            x_batch.shape[1] == 80
        ), f"Sequence length must be 80, got {x_batch.shape[1]}"

        # Expected Target: (Batch, Seq_Len=80)
        assert y_batch.dim() == 2, f"Target must be 2D, got {y_batch.shape}"
        assert (
            y_batch.shape[1] == 80
        ), f"Target sequence length must be 80, got {y_batch.shape[1]}"

        # Verify Feature Count matches Config
        expected_feats = len(Config.FEATURE_COLS)
        assert (
            x_batch.shape[2] == expected_feats
        ), f"Feature count mismatch. Config: {expected_feats}, Batch: {x_batch.shape[2]}"

        print(f"Data verification passed. Batch shape: {x_batch.shape}")

    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # --------------------------------------------------------------------------
    # 3. Model Initialization and Logic Verification
    # --------------------------------------------------------------------------
    print("\n[Step 2] Initializing PM-NC-Net Model...")
    model = PMNCNet().to(Config.DEVICE)

    # Verify Forward Pass
    print("Verifying model forward pass...")
    model.eval()
    with torch.no_grad():
        x_batch = x_batch.to(Config.DEVICE)
        y_pred = model(x_batch)

    # Output should be (Batch, 80)
    assert (
        y_pred.shape == y_batch.shape
    ), f"Model output shape mismatch. Expected {y_batch.shape}, got {y_pred.shape}"
    print("Model forward pass successful.")

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    print(f"\n[Step 3] Starting Training for {Config.EPOCHS} epochs (Debug Mode)...")
    trainer = Trainer(train_loader, val_loader)

    # Run training
    trainer.fit()

    # Verify Checkpoint Creation
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Best model checkpoint was not created at {Config.BEST_MODEL_PATH}"
        )
    print("Training complete. Checkpoint saved.")

    # --------------------------------------------------------------------------
    # 5. Prediction and Submission
    # --------------------------------------------------------------------------
    print("\n[Step 4] Generating Submission...")
    generate_submission(trainer, test_loader, test_ids)

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_FILE}"
        )

    # Verify Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission loaded. Shape: {sub_df.shape}")

    # In DEBUG mode, we slice the test set to 1000 breaths.
    # Each breath has 80 time steps. Total rows should be 80,000.
    expected_rows = 1000 * 80
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Verify Columns
    expected_cols = ["id", "pressure"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Verify values are numeric
    assert pd.api.types.is_numeric_dtype(
        sub_df["pressure"]
    ), "Pressure predictions are not numeric."

    print("\n=== Demo Execution Completed Successfully ===")
    print(f"Output available at: {Config.SUBMISSION_FILE}")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    main()
