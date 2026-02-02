import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.dataset import get_data_loaders
from library.model import RDHNet
from library.utils import compute_metric, seed_everything
from library.train import run_training


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup Configuration
    # We use debug=True to limit data size (100 breaths) and epochs (2) for speed.
    print("\n[Step 1] Initializing Configuration...")
    config = Config(debug=True)

    # Override working directory for this demo to ensure isolation
    config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Update cache paths to be inside the new working dir
    config.CACHE_TRAIN = os.path.join(config.WORKING_DIR, "train_cache.parquet")
    config.CACHE_VAL = os.path.join(config.WORKING_DIR, "val_cache.parquet")
    config.CACHE_TEST = os.path.join(config.WORKING_DIR, "test_cache.parquet")
    config.SCALER_PATH = os.path.join(config.WORKING_DIR, "scaler.joblib")

    config.display()

    # Set random seeds for reproducibility
    seed_everything(config.SEED)

    # 2. Verify Data Loading
    print("\n[Step 2] Verifying Data Loading Pipeline...")
    # This handles loading, feature engineering, scaling, and reshaping
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # Fetch a single batch to verify shapes
    batch = next(iter(train_loader))
    x, u_out, y = batch["x"], batch["u_out"], batch["y"]

    print(f"  Batch X Shape: {x.shape}")
    print(f"  Batch u_out Shape: {u_out.shape}")
    print(f"  Batch y Shape: {y.shape}")

    # Assertions
    # Shape: (Batch_Size, Seq_Len=80, Features)
    assert x.dim() == 3, "Input X must be 3-dimensional"
    assert (
        x.size(0) == config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {config.BATCH_SIZE}, got {x.size(0)}"
    assert x.size(1) == 80, "Sequence length must be 80"
    assert (
        x.size(2) == config.INPUT_DIM
    ), f"Input feature dimension mismatch. Expected {config.INPUT_DIM}, got {x.size(2)}"
    assert u_out.shape == (config.BATCH_SIZE, 80), "u_out shape mismatch"
    assert y.shape == (config.BATCH_SIZE, 80), "Target y shape mismatch"

    print("  -> Data Loading Verified.")

    # 3. Verify Model Architecture
    print("\n[Step 3] Verifying Model Architecture...")
    device = torch.device(config.DEVICE)
    model = RDHNet(config).to(device)

    # Move batch to device
    x_gpu = x.to(device)

    # Forward Pass
    with torch.no_grad():
        output = model(x_gpu)

    print(f"  Model Output Shape: {output.shape}")

    # Assertions
    # Output: (Batch_Size, Seq_Len=80, 1)
    assert output.size() == (config.BATCH_SIZE, 80, 1), "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model produced NaN values"

    print("  -> Model Architecture Verified.")

    # 4. Verify Metric Calculation
    print("\n[Step 4] Verifying Metric Logic (MAE on Inspiratory Phase)...")

    # Case 1: Perfect prediction
    # u_out = 0 (Inspiratory), u_out = 1 (Expiratory)
    # We predict 10 for everything. Target is 10 for Insp, 99 for Exp.
    # Error should be 0 because Exp is ignored.
    dummy_pred = torch.tensor([10.0, 10.0])
    dummy_target = torch.tensor([10.0, 99.0])
    dummy_u_out = torch.tensor([0, 1])

    mae_1 = compute_metric(dummy_pred, dummy_target, dummy_u_out)
    print(f"  Test Case 1 (Perfect Insp, Bad Exp): MAE = {mae_1}")
    assert mae_1 == 0.0, f"Metric failed. Expected 0.0, got {mae_1}"

    # Case 2: Known Error
    # Insp: Pred 10, Target 12 -> Error 2
    # Exp: Pred 10, Target 10 -> Error 0 (Ignored)
    dummy_pred_2 = torch.tensor([10.0, 10.0])
    dummy_target_2 = torch.tensor([12.0, 10.0])
    dummy_u_out_2 = torch.tensor([0, 1])

    mae_2 = compute_metric(dummy_pred_2, dummy_target_2, dummy_u_out_2)
    print(f"  Test Case 2 (Error Insp): MAE = {mae_2}")
    assert abs(mae_2 - 2.0) < 1e-6, f"Metric failed. Expected 2.0, got {mae_2}"

    print("  -> Metric Logic Verified.")

    # 5. Run Full Training Loop
    print("\n[Step 5] Executing Training Loop (Debug Mode)...")
    # This will run for config.EPOCHS (2) on the debug subset (100 breaths)
    # It handles training, validation, saving checkpoints, etc.
    trained_model = run_training(config)

    # 6. Verify Outputs
    print("\n[Step 6] Verifying Training Artifacts...")
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    checkpoint_path = os.path.join(config.WORKING_DIR, "checkpoint.pth")
    scaler_path = config.SCALER_PATH

    if os.path.exists(best_model_path):
        print(f"  -> Found Best Model: {best_model_path}")
    else:
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    if os.path.exists(scaler_path):
        print(f"  -> Found Scaler: {scaler_path}")
    else:
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")

    # 7. Final Inference Check
    print("\n[Step 7] Running Inference on Test Subset...")
    trained_model.eval()
    predictions = []

    with torch.no_grad():
        for batch_data in test_loader:
            x_test = batch_data["x"].to(device)
            out = trained_model(x_test)
            # Flatten predictions
            predictions.extend(out.cpu().numpy().flatten())

    num_preds = len(predictions)
    print(f"  Generated {num_preds} predictions.")

    # In debug mode, we use 100 breaths. Each breath has 80 time steps.
    expected_preds = config.DEBUG_BREATH_COUNT * 80
    assert (
        num_preds == expected_preds
    ), f"Prediction count mismatch. Expected {expected_preds}, got {num_preds}"

    print("  -> Inference Verified.")

    print("\n=== Demonstration Complete: All Systems Go ===")


if __name__ == "__main__":
    main()
