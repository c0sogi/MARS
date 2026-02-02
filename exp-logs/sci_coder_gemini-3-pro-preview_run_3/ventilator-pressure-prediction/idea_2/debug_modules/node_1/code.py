import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.dataset import get_dataloaders
from library.model import HybridCNNLSTM
from library.engine import masked_mae_loss, run_training


def main():
    print("=== Starting Ventilator Pressure Prediction Demo ===")

    # 1. Configuration Setup
    # Modify working directory for this demo to keep it isolated
    Config.WORKING_DIR = "./working/demo_execution/"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")

    # Update cache paths to point to the new working directory
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_cache.npy")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_cache.npy")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_cache.npy")

    # Initialize environment (creates dirs, sets seeds)
    Config.initialize()

    # Explicitly set seeds again to be absolutely sure
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Pipeline Verification
    print("\n--- Verifying Data Pipeline ---")

    # Use debug=True to load only a small subset (1000 breaths) for speed
    # load_cached_data=False ensures we test the feature engineering pipeline
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Fetch one batch to verify shapes
    x_batch, y_batch = next(iter(train_loader))

    print(f"Batch X shape: {x_batch.shape}")
    print(f"Batch Y shape: {y_batch.shape}")

    # Assertions for Data
    # Shape: (Batch_Size, Seq_Len, Num_Features)
    assert x_batch.dim() == 3, "Input tensor must be 3-dimensional"
    assert (
        x_batch.shape[1] == Config.SEQ_LEN
    ), f"Sequence length must be {Config.SEQ_LEN}"
    assert x_batch.shape[2] == len(
        Config.FEATURE_COLS
    ), f"Feature dim must match {len(Config.FEATURE_COLS)}"

    # Shape: (Batch_Size, Seq_Len)
    assert y_batch.dim() == 2, "Target tensor must be 2-dimensional"
    assert y_batch.shape[1] == Config.SEQ_LEN, "Target sequence length mismatch"

    print("Data Pipeline verification passed.")

    # 3. Model Architecture Verification
    print("\n--- Verifying Model Architecture ---")

    device = torch.device(Config.DEVICE)
    model = HybridCNNLSTM().to(device)

    # Move batch to device
    x_batch = x_batch.to(device)

    # Forward pass
    with torch.no_grad():
        y_pred = model(x_batch)

    print(f"Prediction shape: {y_pred.shape}")

    # Assertions for Model
    assert (
        y_pred.shape == y_batch.shape
    ), "Model output shape does not match target shape"

    print("Model Architecture verification passed.")

    # 4. Loss Function Verification
    print("\n--- Verifying Loss Function (Masked MAE) ---")

    # Create synthetic data
    # y_true: [10, 10, 10, 10]
    # y_pred: [12, 12, 15, 15]
    # u_out:  [ 0,  0,  1,  1] (0=Inspiratory, 1=Expiratory)
    # Errors: [ 2,  2,  5,  5]
    # Mask:   [ 1,  1,  0,  0]
    # Valid Errors: [2, 2] -> Mean = 2.0

    dummy_true = torch.tensor([[10.0, 10.0, 10.0, 10.0]], device=device)
    dummy_pred = torch.tensor([[12.0, 12.0, 15.0, 15.0]], device=device)
    dummy_u_out = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device)

    loss = masked_mae_loss(dummy_pred, dummy_true, dummy_u_out)

    print(f"Calculated Loss: {loss.item()}")

    # Assertions for Loss
    expected_loss = 2.0
    assert (
        abs(loss.item() - expected_loss) < 1e-6
    ), f"Loss calculation incorrect. Expected {expected_loss}, got {loss.item()}"

    print("Loss Function verification passed.")

    # 5. End-to-End Training Loop
    print("\n--- Running Training Loop (Demo) ---")

    # Run for 2 epochs to test the loop, validation, and saving mechanics
    # Using the debug loaders created earlier
    run_training(train_loader, val_loader, test_loader, epochs=2, patience=1)

    # 6. Submission Output Verification
    print("\n--- Verifying Submission Output ---")

    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission shape: {submission_df.shape}")
    print(f"Submission columns: {submission_df.columns.tolist()}")

    # Calculate expected rows:
    # Debug mode samples Config.DEBUG_SAMPLE_SIZE breaths (default 1000)
    # But wait, prepare_dataset samples breaths.
    # The test set might have fewer than DEBUG_SAMPLE_SIZE unique breaths if the file is small,
    # but here test.csv is large.
    # In debug mode, prepare_dataset samples 1000 breaths.
    # 1000 breaths * 80 time_steps = 80,000 rows.

    expected_rows = Config.DEBUG_SAMPLE_SIZE * Config.SEQ_LEN

    # Note: If the test set has fewer breaths than DEBUG_SAMPLE_SIZE, it would be that length.
    # Given the dataset info, test set is large enough.

    assert (
        submission_df.shape[0] == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {submission_df.shape[0]}"

    assert (
        "id" in submission_df.columns and "pressure" in submission_df.columns
    ), "Submission missing required columns"

    # Check for NaNs
    assert not submission_df.isnull().values.any(), "Submission contains NaN values"

    print("Submission verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
