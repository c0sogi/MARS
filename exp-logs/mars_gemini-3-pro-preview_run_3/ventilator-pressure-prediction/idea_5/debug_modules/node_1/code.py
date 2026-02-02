import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, compute_metric, get_device
from library.data import get_dataloaders
from library.model import CAPNet, masked_mae_loss
from library.train import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Overrides
    # We override Config to run a fast, lightweight demonstration.
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set a specific working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Patch Config class attributes
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.MODEL_PATH = os.path.join(DEMO_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce Model Complexity for speed
    Config.LSTM_HIDDEN_SIZE = 32
    Config.LSTM_NUM_LAYERS = 1
    Config.TCN_CHANNELS = [16, 32]  # Reduced depth and width
    Config.FC_HIDDEN_SIZE = 32

    # Reduce Data/Training parameters
    Config.DEBUG = True
    Config.DEBUG_SIZE = 500  # Use only 500 breaths
    Config.BATCH_SIZE = 16
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure directories exist
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Pipeline Verification
    print("\n[2] Verifying Data Pipeline...")

    # Force clean start to test processing logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch to verify shapes
    x_batch, u_out_batch, y_batch = next(iter(train_loader))

    # Expected shapes:
    # x: (Batch, Seq_Len=80, Input_Dim=9)
    # u_out: (Batch, Seq_Len=80)
    # y: (Batch, Seq_Len=80)
    print(
        f"Batch shapes -> X: {x_batch.shape}, u_out: {u_out_batch.shape}, y: {y_batch.shape}"
    )

    assert x_batch.shape == (
        Config.BATCH_SIZE,
        Config.BREATH_LEN,
        Config.INPUT_DIM,
    ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, Config.BREATH_LEN, Config.INPUT_DIM)}, got {x_batch.shape}"
    assert y_batch.shape == (
        Config.BATCH_SIZE,
        Config.BREATH_LEN,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, Config.BREATH_LEN)}, got {y_batch.shape}"
    assert u_out_batch.shape == (
        Config.BATCH_SIZE,
        Config.BREATH_LEN,
    ), f"u_out shape mismatch. Expected {(Config.BATCH_SIZE, Config.BREATH_LEN)}, got {u_out_batch.shape}"

    print("Data Pipeline verification passed.")

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture...")

    model = CAPNet().to(device)

    # Move batch to device
    x_batch = x_batch.to(device)

    # Forward pass
    y_pred = model(x_batch)

    print(f"Model Output Shape: {y_pred.shape}")

    # Verify output shape matches target shape (Batch, Seq_Len)
    assert y_pred.shape == (
        Config.BATCH_SIZE,
        Config.BREATH_LEN,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.BREATH_LEN)}, got {y_pred.shape}"

    print("Model Architecture verification passed.")

    # 4. Metric Logic Verification
    print("\n[4] Verifying Metric Logic (Inspiratory Phase Only)...")

    # Create dummy data
    # Case:
    # u_out = 0 (Inspiratory) -> Error should count
    # u_out = 1 (Expiratory) -> Error should be ignored

    dummy_pred = torch.tensor([10.0, 10.0, 10.0, 10.0])
    dummy_true = torch.tensor([12.0, 12.0, 20.0, 20.0])
    dummy_u_out = torch.tensor([0.0, 0.0, 1.0, 1.0])  # Last two are expiratory

    # Expected MAE:
    # Indices 0, 1 are relevant (u_out=0). Errors: |10-12|=2, |10-12|=2. Mean = 2.0.
    # Indices 2, 3 are ignored.

    # Test compute_metric (numpy/tensor input)
    mae_score = compute_metric(dummy_pred, dummy_true, dummy_u_out)
    print(f"Computed MAE: {mae_score}")

    assert (
        abs(mae_score - 2.0) < 1e-6
    ), f"Metric calculation failed. Expected 2.0, got {mae_score}"

    # Test masked_mae_loss (tensor input, differentiable)
    loss = masked_mae_loss(dummy_pred, dummy_true, dummy_u_out)
    print(f"Computed Loss: {loss.item()}")

    assert (
        abs(loss.item() - 2.0) < 1e-6
    ), f"Loss calculation failed. Expected 2.0, got {loss.item()}"

    print("Metric Logic verification passed.")

    # 5. Full Training Loop Execution
    print("\n[5] Executing Full Training Loop (Short Run)...")

    # We use the run_training function from library.train
    # Note: We pass clean_start=False because we already processed data in step 2
    # and we want to reuse the cache in DEMO_DIR.
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=1e-3,
        debug=True,
        clean_start=False,
        load_cache=True,
    )

    # 6. Submission Verification
    print("\n[6] Verifying Submission Artifacts...")

    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Check submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{sub_df.head()}")

    assert (
        "id" in sub_df.columns and "pressure" in sub_df.columns
    ), "Submission file missing required columns 'id' or 'pressure'"

    # Check row count
    # In debug mode, we used 500 breaths.
    # Test split is roughly 20% of total if random split, but here we used
    # train/val/test split from metadata.
    # The debug logic in data.py slices the first N unique breaths from EACH split.
    # So test set has 500 breaths * 80 steps = 40,000 rows.
    expected_rows = Config.DEBUG_SIZE * Config.BREATH_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    print("Submission verification passed.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
