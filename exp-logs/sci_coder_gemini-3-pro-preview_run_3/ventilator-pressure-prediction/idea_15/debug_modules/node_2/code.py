import os
import shutil
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_mae, get_device
from library.dataset import prepare_data
from library.model import FMDHNet
from library.trainer import Trainer


def run_demo():
    print("=== Starting Demonstration of Ventilator Pressure Prediction Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Define a separate working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Modify Config attributes globally
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Enable Debug mode to use a tiny subset of data (e.g., 200 breaths)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200

    # Reduce training parameters
    Config.BATCH_SIZE = 32
    Config.EPOCHS = 2
    Config.EARLY_STOPPING_PATIENCE = 1

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2] initializing Data Pipeline (Feature Engineering & Loading)...")

    # prepare_data handles loading, feature engineering, scaling, and caching
    train_loader, val_loader, test_loader, test_ids = prepare_data(
        batch_size=Config.BATCH_SIZE, load_cached_data=False
    )

    # Verify Train Loader
    x_batch, y_batch = next(iter(train_loader))
    print(f"Train Batch X Shape: {x_batch.shape}")  # Expected: (32, 80, 14)
    print(f"Train Batch Y Shape: {y_batch.shape}")  # Expected: (32, 80)

    # Assertions
    assert x_batch.dim() == 3, "Input batch must be 3D (Batch, Seq, Feat)"
    assert (
        x_batch.shape[1] == Config.SEQ_LEN
    ), f"Sequence length must be {Config.SEQ_LEN}"
    assert (
        x_batch.shape[2] == Config.INPUT_DIM
    ), f"Feature count must be {Config.INPUT_DIM}"
    assert y_batch.dim() == 2, "Target batch must be 2D (Batch, Seq)"

    # Verify Test IDs
    print(f"Test IDs Shape: {test_ids.shape}")
    assert test_ids.ndim == 2, "Test IDs should be 2D array"
    assert test_ids.shape[1] == Config.SEQ_LEN, "Test IDs seq len mismatch"

    print("Data Pipeline verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Initializing and Verifying FMDH-Net Model...")

    device = get_device()
    model = FMDHNet().to(device)

    # Move batch to device
    x_batch = x_batch.to(device)

    # Forward pass
    with torch.no_grad():
        preds = model(x_batch)

    print(f"Model Output Shape: {preds.shape}")

    # Assertions
    # Output should be (Batch, Seq_Len, 1)
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.SEQ_LEN, 1)}, got {preds.shape}"

    print("Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 4. Metric Logic Verification (Unit Test)
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Metric Calculation (MAE on Inspiratory Phase)...")

    # Create synthetic data
    # Case: 2 time steps.
    # Step 0: u_out=0 (Inspiratory) -> Should be counted.
    # Step 1: u_out=1 (Expiratory) -> Should be IGNORED.

    # Preds: [10.0, 100.0]
    # Targets: [12.0, 50.0]
    # u_out: [0, 1]

    # Expected MAE: |10 - 12| = 2.0. The error |100 - 50| = 50 should be ignored.

    dummy_preds = torch.tensor([10.0, 100.0])
    dummy_targets = torch.tensor([12.0, 50.0])
    dummy_u_out = torch.tensor([0, 1])  # 0=Inspiratory, 1=Expiratory

    calculated_mae = compute_mae(dummy_preds, dummy_targets, dummy_u_out)

    print(f"Calculated MAE: {calculated_mae}")
    assert (
        abs(calculated_mae - 2.0) < 1e-6
    ), f"Metric logic failed. Expected 2.0, got {calculated_mae}"

    print("Metric logic verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Loop (Fit)...")

    trainer = Trainer(model)

    # Run training (2 epochs as configured)
    trainer.fit(train_loader, val_loader)

    # Verify model checkpoint exists
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Best model checkpoint was not saved."
    print(f"Checkpoint verified at: {Config.MODEL_SAVE_PATH}")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6] Executing Inference (Predict)...")

    trainer.predict(test_loader, test_ids)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Loaded. Shape: {sub_df.shape}")
    print(f"Columns: {list(sub_df.columns)}")

    # Assertions on submission
    assert (
        "id" in sub_df.columns and "pressure" in sub_df.columns
    ), "Submission missing required columns."
    assert len(sub_df) > 0, "Submission file is empty."

    # Check if number of rows matches test_ids count
    expected_rows = test_ids.size
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    print("Inference and submission verification passed.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
