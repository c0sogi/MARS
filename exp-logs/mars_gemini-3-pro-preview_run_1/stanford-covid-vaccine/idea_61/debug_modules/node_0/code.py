import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, mcrmse
from library.data import get_dataloaders
from library.model import StabilizedWideBiLSTM
from library.engine import train_model, generate_submission
from library.loss import MaskedMSELoss


def run_demo():
    print("--- Starting RNA Degradation Prediction Demo ---")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Modify Config for a fast demonstration run
    Config.PROJECT_NAME = "demo_run"
    Config.WORKING_DIR = f"./working/{Config.PROJECT_NAME}"
    Config.SUBMISSION_DIR = f"{Config.WORKING_DIR}/submission"

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Update file paths to point to the demo directory
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Use distinct cache files for the demo to avoid conflicts with full runs
    Config.CACHE_TRAIN_PATH = os.path.join(Config.WORKING_DIR, "cached_train.pt")
    Config.CACHE_VAL_PATH = os.path.join(Config.WORKING_DIR, "cached_val.pt")
    Config.CACHE_TEST_PATH = os.path.join(Config.WORKING_DIR, "cached_test.pt")

    # Set hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Only use 50 samples
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n--- Loading Data ---")
    # Force reprocessing by setting load_cached_data=False initially or ensuring cache doesn't exist
    # Since we changed cache paths to a new dir, they won't exist.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,  # Force processing of the debug subset
        debug=Config.DEBUG,
    )

    # Verify DataLoaders
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Verify Batch Shapes
    sample_inputs, sample_targets = next(iter(train_loader))
    print(f"Sample Input Shape: {sample_inputs.shape} (Batch, Seq_Len, 3)")
    print(f"Sample Target Shape: {sample_targets.shape} (Batch, Seq_Len, Num_Targets)")

    # Assertions
    assert sample_inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
    ), "Incorrect input shape"
    assert sample_targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), "Incorrect target shape"
    assert not torch.isnan(sample_inputs).any(), "Inputs contain NaNs"
    # Targets might be 0-padded, which is fine.

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass Check
    # --------------------------------------------------------------------------
    print("\n--- Initializing Model ---")
    device = Config.DEVICE
    model = StabilizedWideBiLSTM().to(device)

    # Run a dummy forward pass
    print("Running dummy forward pass...")
    model.eval()
    with torch.no_grad():
        dummy_out = model(sample_inputs.to(device))

    print(f"Model Output Shape: {dummy_out.shape}")

    # Assertions
    assert dummy_out.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)}, got {dummy_out.shape}"

    # --------------------------------------------------------------------------
    # 4. Loss Function Check
    # --------------------------------------------------------------------------
    print("\n--- Checking Loss Function ---")
    criterion = MaskedMSELoss()
    loss = criterion(dummy_out, sample_targets.to(device))
    print(f"Initial Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"

    # --------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n--- Starting Training Loop ---")
    # We use the provided train_model function which handles the loop, validation, and saving
    train_model(model, train_loader, val_loader)

    # Verify model file was saved
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved."
    print("Training completed successfully.")

    # --------------------------------------------------------------------------
    # 6. Submission Generation
    # --------------------------------------------------------------------------
    print("\n--- Generating Submission ---")
    # Generate submission using the best model
    generate_submission(model, test_loader)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print("Submission Head:")
    print(df_sub.head())

    # Assertions on Submission
    # Expected rows: Number of test samples * Seq Length
    # In debug mode, we loaded 50 test samples (or min(50, total_test)).
    # Total test samples is 240. Debug subset is 50.
    n_test_samples = min(Config.DEBUG_SUBSET_SIZE, 240)
    expected_rows = n_test_samples * Config.SEQ_LEN

    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Column mismatch. Got {list(df_sub.columns)}"

    # Check that unscored columns are 0.0 as per generate_submission logic
    assert (df_sub["deg_pH10"] == 0.0).all(), "deg_pH10 should be 0.0"
    assert (df_sub["deg_50C"] == 0.0).all(), "deg_50C should be 0.0"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
