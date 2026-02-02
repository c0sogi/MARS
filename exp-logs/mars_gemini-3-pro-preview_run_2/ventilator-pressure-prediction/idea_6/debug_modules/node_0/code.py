import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import prepare_data
from library.model import WideDeepBiLSTM
from library.train import run_training, weighted_l1_loss


def main():
    # 1. Setup and Configuration
    # ==========================
    print("--- 1. Initialization and Configuration ---")
    seed_everything(42)

    # Define a Demo Configuration to run quickly
    class DemoConfig(Config):
        # Use a separate directory for demo artifacts
        WORKING_DIR = "./working/demo_execution"

        # Reduce model size for speed
        LSTM_HIDDEN_SIZE = 64
        LSTM_LAYERS = 2
        MLP_HIDDEN_SIZE = 32
        MLP_LAYERS = 2

        # Training hyperparameters for demo
        EPOCHS = 2
        BATCH_SIZE = 16
        LEARNING_RATE = 1e-3

        # Data subset size
        LIMIT_BREATHS = 100  # Load only 100 breaths

    config = DemoConfig()

    # Clean up previous demo run if exists
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # 2. Data Pipeline Verification
    # =============================
    print("\n--- 2. Verifying Data Pipeline ---")

    # Load a small subset of training data
    # debug=True appends '_debug' to cache filenames to avoid conflicts
    train_loader = prepare_data(
        split="train",
        config=config,
        load_cached_data=False,
        debug=True,
        limit_breaths=config.LIMIT_BREATHS,
    )

    # Fetch a single batch
    batch = next(iter(train_loader))
    X, y, u_out = batch["X"], batch["y"], batch["u_out"]

    print(f"Batch X shape: {X.shape}")  # Expected: (16, 80, Features)
    print(f"Batch y shape: {y.shape}")  # Expected: (16, 80)
    print(f"Batch u_out shape: {u_out.shape}")  # Expected: (16, 80)

    # Assertions to verify data integrity
    assert X.ndim == 3, "Input X must be 3D: (Batch, Seq, Features)"
    assert (
        X.shape[0] == config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {config.BATCH_SIZE}, got {X.shape[0]}"
    assert X.shape[1] == 80, "Sequence length must be exactly 80 time steps per breath"
    assert y.shape == (config.BATCH_SIZE, 80), "Target y shape mismatch"
    assert u_out.shape == (config.BATCH_SIZE, 80), "u_out shape mismatch"

    # Check if u_out is binary (0 or 1)
    unique_u_out = torch.unique(u_out)
    assert torch.all(
        (unique_u_out == 0) | (unique_u_out == 1)
    ), "u_out must contain only binary values (0, 1)"

    print("Data Pipeline Verified Successfully.")

    # 3. Model Architecture Verification
    # ==================================
    print("\n--- 3. Verifying Model Architecture ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X.shape[-1]

    # Instantiate the Wide & Deep BiLSTM model
    model = WideDeepBiLSTM(input_dim=input_dim, config=config).to(device)

    # Move data to device
    X_device = X.to(device)
    y_device = y.to(device)
    u_out_device = u_out.to(device)

    # Forward Pass
    pred = model(X_device)
    print(f"Prediction shape: {pred.shape}")

    # Assertions for output
    assert pred.shape == (config.BATCH_SIZE, 80), "Model output shape mismatch"
    assert not torch.isnan(pred).any(), "Model produced NaN values in forward pass"

    # Loss Calculation
    loss = weighted_l1_loss(pred, y_device, u_out_device, config)
    print(f"Calculated Loss: {loss.item():.6f}")
    assert loss.item() >= 0, "Loss cannot be negative"

    # Backward Pass (Gradient Check)
    loss.backward()

    # Check for valid gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            assert not torch.isnan(param.grad).any(), f"NaN gradient detected in {name}"

    assert has_grad, "No gradients computed. Backward pass failed."
    print("Model Architecture and Gradient Flow Verified.")

    # 4. Training Loop Verification
    # =============================
    print("\n--- 4. Verifying Training Loop ---")

    # Run the full training routine (for 2 epochs on a tiny dataset)
    # This verifies the integration of Dataset, Model, Optimizer, Scheduler, and MetricMonitor
    best_mae = run_training(
        config=config, debug=True, limit_breaths=config.LIMIT_BREATHS
    )

    print(f"Training run completed. Best MAE: {best_mae:.6f}")

    # Check if the best model was saved
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(
        best_model_path
    ), "best_model.pth was not saved after training"
    print("Training Loop Verified.")

    # 5. Inference Verification
    # =========================
    print("\n--- 5. Verifying Inference Logic ---")

    # Load the saved model state
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Run inference on the sample batch
    with torch.no_grad():
        val_pred = model(X_device)

    # Basic sanity checks on predictions
    mae = torch.abs(val_pred - y_device).mean().item()
    print(f"Inference MAE on sample batch: {mae:.6f}")

    assert val_pred.shape == (config.BATCH_SIZE, 80), "Inference output shape mismatch"

    print("Inference Logic Verified.")
    print("\n=== All Demonstrations Passed Successfully ===")


if __name__ == "__main__":
    main()
