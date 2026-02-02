import os
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the python path for library imports
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config, seed_everything
from library.features import FeatureEngineer
from library.dataset import get_data_loaders
from library.model import VentilatorModel
from library.utils import get_device, compute_mae
from library.train import train_fn, valid_fn, inference_fn


def run_demo():
    print("=== Starting Ventilator Pressure Prediction Demo ===")

    # 1. Setup & Configuration Override for Speed
    # We modify the Config class directly to run a fast debug session
    print("Configuring for fast debug run...")
    Config.DEBUG = True
    Config.DEBUG_BREATHS = 200  # Use only 200 breaths for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16  # Smaller batch size for debug
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Pipeline Demonstration
    print("\n--- 1. Testing Data Pipeline (Feature Engineering & Loading) ---")
    # This triggers FeatureEngineer internally, processes the debug split, and returns loaders
    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=False)

    # Verify DataLoaders
    print("Verifying DataLoader outputs...")
    batch = next(iter(train_loader))

    # Check keys
    assert "x" in batch, "Batch missing 'x'"
    assert "y" in batch, "Batch missing 'y'"
    assert "u_out" in batch, "Batch missing 'u_out'"

    # Check shapes
    # x: (Batch, Seq_Len, Input_Dim)
    # y: (Batch, Seq_Len)
    # u_out: (Batch, Seq_Len)
    batch_size = batch["x"].size(0)
    seq_len = batch["x"].size(1)
    input_dim = batch["x"].size(2)

    assert (
        batch_size == Config.BATCH_SIZE
    ), f"Expected batch size {Config.BATCH_SIZE}, got {batch_size}"
    assert (
        seq_len == Config.SEQ_LEN
    ), f"Expected seq len {Config.SEQ_LEN}, got {seq_len}"
    assert (
        input_dim == Config.INPUT_DIM
    ), f"Expected input dim {Config.INPUT_DIM}, got {input_dim}"

    print(f"Batch shapes verified: X={batch['x'].shape}, Y={batch['y'].shape}")

    # 3. Metric Verification (Unit Test)
    print("\n--- 2. Verifying Metric Logic (MAE with Masking) ---")
    # Construct a manual case
    # u_out = 0 (inspiratory, included), u_out = 1 (expiratory, excluded)
    preds = np.array([10, 20, 30, 40])
    targets = np.array([12, 20, 100, 200])  # Errors: 2, 0, 70, 160
    u_out = np.array([0, 0, 1, 1])  # Mask: Include, Include, Exclude, Exclude

    # Expected MAE: (|10-12| + |20-20|) / 2 = (2 + 0) / 2 = 1.0
    # The large errors in u_out=1 region should be ignored
    calculated_mae = compute_mae(preds, targets, u_out)
    print(f"Calculated MAE: {calculated_mae}")

    assert (
        abs(calculated_mae - 1.0) < 1e-6
    ), f"Metric verification failed. Expected 1.0, got {calculated_mae}"
    print("Metric logic verified.")

    # 4. Model Instantiation & Training Loop Demo
    print("\n--- 3. Testing Model & Training Loop ---")
    model = VentilatorModel()
    model.to(device)

    # Check model forward pass structure
    dummy_x = batch["x"].to(device)
    # Training mode returns tuple (final, aux)
    model.train()
    out_train = model(dummy_x)
    assert isinstance(
        out_train, tuple
    ), "Model in train mode should return tuple (final, aux)"
    assert len(out_train) == 2
    assert out_train[0].shape == (
        batch_size,
        seq_len,
    ), f"Output shape mismatch. Got {out_train[0].shape}"

    # Eval mode returns tensor
    model.eval()
    out_eval = model(dummy_x)
    assert torch.is_tensor(out_eval), "Model in eval mode should return single tensor"

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.LR, total_steps=total_steps, pct_start=0.3
    )

    print(f"Running training for {Config.EPOCHS} epoch(s) on debug subset...")
    train_loss = train_fn(model, train_loader, optimizer, scheduler, device)
    print(f"Train Loss: {train_loss:.4f}")

    assert np.isfinite(train_loss), "Training loss is NaN or Infinite"

    print("Running validation...")
    val_mae = valid_fn(model, val_loader, device)
    print(f"Validation MAE: {val_mae:.4f}")

    # Save model for inference step
    model_path = os.path.join(Config.WORKING_DIR, "model.pth")
    torch.save(model.state_dict(), model_path)
    assert os.path.exists(model_path), "Model file was not saved"

    # 5. Inference Demo
    print("\n--- 4. Testing Inference & Submission ---")
    # Reload model to verify loading works
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    inference_fn(model, test_loader, device)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with shape: {sub_df.shape}")
    print(f"First few rows:\n{sub_df.head()}")

    # Check columns
    assert (
        "id" in sub_df.columns and "pressure" in sub_df.columns
    ), "Submission missing required columns"

    # Check that we have predictions for the debug subset of test data
    # Since we used DEBUG mode, the test set is also sliced.
    # We just verify it's not empty.
    assert len(sub_df) > 0, "Submission file is empty"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
