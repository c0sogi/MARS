import os
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.data_utils import get_data_loaders
from library.model import MSDHNet
from library.train_utils import masked_mae_loss, run_training, set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Ventilator Pressure Prediction: Library Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Modify Config global state to run a fast, lightweight version
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16  # Smaller batch size for debug data
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = Config.WORKING_DIR
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Pipeline (get_data_loaders)...")

    # Load data in debug mode (small subset of breaths)
    # load_cached_data=False forces processing from scratch to demonstrate feature engineering
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        load_cached_data=False, debug=True
    )

    # Fetch a single batch to verify shapes
    x_batch, y_batch = next(iter(train_loader))

    print(f"    Batch X shape: {x_batch.shape}")
    print(f"    Batch y shape: {y_batch.shape}")

    # Assertions for Data
    expected_x_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM)
    expected_y_shape = (Config.BATCH_SIZE, Config.SEQ_LEN)

    # Note: The last batch might be smaller, but with 100 breaths and batch 16,
    # the first batch should be full size.
    assert (
        x_batch.shape == expected_x_shape
    ), f"Expected X shape {expected_x_shape}, got {x_batch.shape}"
    assert (
        y_batch.shape == expected_y_shape
    ), f"Expected y shape {expected_y_shape}, got {y_batch.shape}"

    print("    -> Data shapes verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Model Architecture (MSDHNet)...")

    device = torch.device(Config.DEVICE)
    model = MSDHNet().to(device)

    # Move batch to device
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    # Forward pass
    y_pred = model(x_batch)

    print(f"    Prediction shape: {y_pred.shape}")

    # Assertions for Model
    assert (
        y_pred.shape == y_batch.shape
    ), f"Model output shape mismatch. Expected {y_batch.shape}, got {y_pred.shape}"
    assert not torch.isnan(y_pred).any(), "Model produced NaN predictions"

    print("    -> Forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Loss Function (masked_mae_loss)...")

    # Extract u_out (Expiratory valve) from input features
    # According to Config.FEATURE_COLS, 'u_out' is at index 1
    u_out = x_batch[:, :, 1]

    loss = masked_mae_loss(y_pred, y_batch, u_out)

    print(f"    Calculated Loss: {loss.item():.6f}")

    # Assertions for Loss
    assert loss.item() >= 0, "Loss cannot be negative"
    assert isinstance(loss, torch.Tensor), "Loss must be a torch Tensor"

    print("    -> Loss calculation verified.")

    # -------------------------------------------------------------------------
    # 5. Full Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Executing Full Training Pipeline (run_training)...")

    # We use load_cached_data=True here because we generated the cache in step [2]
    # This simulates the standard workflow where data is preprocessed once.
    best_loss = run_training(load_cached_data=True, debug=True)

    print(f"    Best Validation Loss: {best_loss:.6f}")

    # -------------------------------------------------------------------------
    # 6. Submission Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Submission Artifacts...")

    # Check if model checkpoint exists
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint not found!"
    print(f"    Checkpoint found at: {Config.MODEL_SAVE_PATH}")

    # Check if submission file exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found!"

    # Validate submission content
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission shape: {submission_df.shape}")
    print(f"    First 3 rows:\n{submission_df.head(3)}")

    # Assertions for Submission
    # In debug mode, we process 50 test breaths.
    # Each breath has Config.SEQ_LEN (80) time steps.
    # Total rows should be 50 * 80 = 4000.
    # However, get_data_loaders debug logic slices unique breath_ids.
    # Let's verify against the test_ids length we got earlier.

    expected_rows = len(test_ids)
    assert (
        len(submission_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(submission_df)}"

    assert list(submission_df.columns) == [
        "id",
        "pressure",
    ], f"Invalid columns: {submission_df.columns}"

    print("    -> Submission format verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
