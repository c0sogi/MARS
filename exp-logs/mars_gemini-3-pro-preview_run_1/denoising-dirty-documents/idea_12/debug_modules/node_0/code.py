import os
import sys
import torch
import numpy as np
import pandas as pd

# Import provided library modules
from library.config import Config
from library.utils import set_seed, calculate_rmse, create_submission_file
from library.model import UNet
from library.train import train_ensemble
from library.inference import predict_and_save


def run_demo():
    print("--- Starting Library Usage Demonstration ---")

    # ==========================================
    # 1. Configuration Override for Speed/Demo
    # ==========================================
    print("\n[1] Configuring environment for demo run...")

    # Set a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Optimize hyperparameters for a fast demonstration
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.ENSEMBLE_SEEDS = [42]  # Use a single seed instead of the full ensemble
    Config.BATCH_SIZE = 4  # Reduced batch size
    Config.NUM_WORKERS = 2  # Minimal workers

    # Initialize the directories based on new config
    Config.initialize()

    # Set global seed for reproducibility of this script
    set_seed(42)

    # ==========================================
    # 2. Utility Function Verification
    # ==========================================
    print("\n[2] Verifying utility functions...")

    # Test RMSE Calculation
    y_true = np.array([0.0, 0.5, 1.0])
    y_pred_perfect = np.array([0.0, 0.5, 1.0])
    y_pred_off = np.array([1.0, 0.5, 0.0])

    rmse_perfect = calculate_rmse(y_true, y_pred_perfect)
    rmse_off = calculate_rmse(y_true, y_pred_off)

    assert rmse_perfect == 0.0, f"Expected RMSE 0.0, got {rmse_perfect}"
    # MSE = ((1-0)^2 + (0.5-0.5)^2 + (0-1)^2) / 3 = (1 + 0 + 1) / 3 = 2/3
    # RMSE = sqrt(0.666...) ~= 0.816
    expected_rmse_off = np.sqrt(2 / 3)
    assert np.isclose(
        rmse_off, expected_rmse_off
    ), f"Expected RMSE {expected_rmse_off}, got {rmse_off}"

    print("    - calculate_rmse: Passed")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    device = Config.DEVICE
    model = UNet(n_channels=Config.IN_CHANNELS, n_classes=Config.OUT_CHANNELS).to(
        device
    )

    # Create a dummy input tensor: (Batch, Channel, Height, Width)
    # Dimensions chosen to be divisible by 16 (2^4) for the 4-level U-Net
    dummy_input = torch.randn(2, 1, 128, 128).to(device)

    # Perform forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify output shape matches input shape (segmentation/denoising task)
    assert (
        output.shape == dummy_input.shape
    ), f"Shape mismatch! Input: {dummy_input.shape}, Output: {output.shape}"

    print(f"    - UNet Forward Pass: Passed (Input/Output shape: {output.shape})")

    # ==========================================
    # 4. Training Pipeline Demonstration
    # ==========================================
    print("\n[4] Running Training Pipeline (1 Epoch)...")

    # This function handles data loading, caching, and the training loop
    # It uses the Config parameters we overrode earlier
    train_ensemble()

    # Verify that the model checkpoint was saved
    expected_model_path = Config.get_model_path(42)
    if not os.path.exists(expected_model_path):
        raise FileNotFoundError(
            f"Training failed to save model to {expected_model_path}"
        )

    print(f"    - Training complete. Model saved to: {expected_model_path}")

    # ==========================================
    # 5. Inference Pipeline Demonstration
    # ==========================================
    print("\n[5] Running Inference Pipeline...")

    # This function loads the model(s), runs TTA, and generates the submission CSV
    predict_and_save()

    # Verify submission file creation
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Inference failed to create submission file at {Config.SUBMISSION_FILE}"
        )

    # Verify submission file content format
    df = pd.read_csv(Config.SUBMISSION_FILE)

    # Check columns
    assert (
        "id" in df.columns and "value" in df.columns
    ), "Submission file missing required columns."

    # Check that we have rows (test set size should be > 0)
    assert len(df) > 0, "Submission file is empty."

    # Check value range (should be between 0 and 1)
    assert (
        df["value"].min() >= 0 and df["value"].max() <= 1
    ), "Pixel values out of range [0, 1]."

    print(f"    - Inference complete. Submission generated with {len(df)} rows.")
    print(f"    - File location: {Config.SUBMISSION_FILE}")

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
