import os
import shutil
import torch
import numpy as np
import pandas as pd

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_mae, get_device
from library.model import NCPNet
from library.data_loader import get_data_loaders
from library.trainer import Trainer


def run_demo():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    print("Configuring runtime environment...")

    # Define a separate working directory for this demo
    DEMO_WORKING_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Patch the Config class to use the demo directory and debug settings
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 200  # Use only 200 breaths for speed
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.FORCE_REGENERATE_CACHE = True  # Ensure we process the subset

    # Manually update cache paths since they were defined at import time
    Config.CACHE_TRAIN_X = os.path.join(DEMO_WORKING_DIR, "train_x.npy")
    Config.CACHE_TRAIN_Y = os.path.join(DEMO_WORKING_DIR, "train_y.npy")
    Config.CACHE_VAL_X = os.path.join(DEMO_WORKING_DIR, "val_x.npy")
    Config.CACHE_VAL_Y = os.path.join(DEMO_WORKING_DIR, "val_y.npy")
    Config.CACHE_TEST_X = os.path.join(DEMO_WORKING_DIR, "test_x.npy")
    Config.CACHE_TEST_IDS = os.path.join(DEMO_WORKING_DIR, "test_ids.npy")
    Config.CACHE_SCALER = os.path.join(DEMO_WORKING_DIR, "scaler_stats.npz")

    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Logic Verification: Metric Calculation
    # -------------------------------------------------------------------------
    print("\nVerifying Metric Logic (compute_mae)...")

    # Synthetic Data
    # Case: 2 samples.
    # Sample 1: u_out=0 (Inspiratory), True=10, Pred=12 -> Error=2
    # Sample 2: u_out=1 (Expiratory),  True=10, Pred=100 -> Error=90 (Should be ignored)
    y_true_syn = torch.tensor([10.0, 10.0])
    y_pred_syn = torch.tensor([12.0, 100.0])
    u_out_syn = torch.tensor([0.0, 1.0])

    mae = compute_mae(y_pred_syn, y_true_syn, u_out_syn)

    print(f"Calculated MAE: {mae}")
    # Expected MAE is 2.0 (only the first sample counts)
    if abs(mae - 2.0) > 1e-6:
        raise AssertionError(f"Metric verification failed. Expected 2.0, got {mae}")
    print("Metric logic verified.")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\nInitializing Data Loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(debug=True)

    # Verify Train Loader
    batch = next(iter(train_loader))
    x, y, u_out = batch["x"], batch["y"], batch["u_out"]

    print(f"Batch Shapes -> X: {x.shape}, Y: {y.shape}, u_out: {u_out.shape}")

    # Assertions for shape
    # Shape: (Batch, Seq_Len, Features)
    assert x.shape[0] == Config.BATCH_SIZE, "Incorrect batch size"
    assert x.shape[1] == Config.MAX_SEQ_LEN, "Incorrect sequence length"
    assert x.shape[2] == len(Config.FEATURE_COLS), "Incorrect feature count"
    assert y.shape == (
        Config.BATCH_SIZE,
        Config.MAX_SEQ_LEN,
        1,
    ), "Incorrect target shape"

    print("Data Loaders verified.")

    # -------------------------------------------------------------------------
    # 4. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\nInitializing Model...")
    model = NCPNet()
    device = get_device()
    model.to(device)

    # Move batch to device for testing
    x_dev = x.to(device)

    print("Running dummy forward pass...")
    with torch.no_grad():
        out = model(x_dev)

    print(f"Model Output Shape: {out.shape}")
    assert out.shape == (
        Config.BATCH_SIZE,
        Config.MAX_SEQ_LEN,
        1,
    ), "Model output shape mismatch"
    print("Model architecture verified.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\nStarting Training Loop (1 Epoch)...")
    trainer = Trainer(model)

    # Fit the model
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Check if model was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise AssertionError("Training finished but best_model.pth was not saved.")

    print(f"Training complete. Model saved to {best_model_path}")

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\nRunning Inference on Test Set...")
    # Load the best model first (good practice check)
    trainer.load_best_model()

    predictions = trainer.predict(test_loader)

    print(f"Predictions shape: {predictions.shape}")

    # Calculate expected number of predictions
    # In debug mode, we took Config.DEBUG_SAMPLES breaths for test set too
    # Total predictions = Num_Breaths * Seq_Len
    expected_preds = Config.DEBUG_SAMPLES * Config.MAX_SEQ_LEN

    # Note: process_split in data_loader handles edge cases, but with clean division:
    assert (
        predictions.size == expected_preds
    ), f"Prediction count mismatch. Expected {expected_preds}, got {predictions.size}"

    print("Inference verified.")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    run_demo()
