import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, calculate_rmse
from library.dataset import DenoisingDataset
from library.model import CoRes2NetUNet
from library.train import run_training
from library.inference import create_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Denoising Pipeline Demonstration ===")

    # 1. Configuration Override for Fast Demonstration
    # We modify the Config class attributes at runtime to adapt to the demo constraints
    # (Speed, specific working directory) without modifying the source file.
    print("\n[1] Configuring Environment...")

    # Set paths for demo outputs
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Important: Update paths that were defined using the original WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_test.csv")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "model.pth")

    # Reduce workload for speed
    Config.PATCHES_PER_IMAGE = 5  # Reduce from 100 to 5 for quick epoch
    Config.NUM_WORKERS = 4  # Use sufficient workers

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(42)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print("Configuration updated for fast execution.")

    # 2. Dataset Verification
    print("\n[2] Verifying Dataset Components...")

    # Test Train Dataset (Patch-based)
    # load_cached_data=False ensures we test the raw image processing logic at least once
    train_ds = DenoisingDataset(mode="train", load_cached_data=False)
    print(f"Training Dataset initialized. Total samples (patches): {len(train_ds)}")

    if len(train_ds) > 0:
        noisy_patch, clean_patch = train_ds[0]
        print(
            f"Sample Train Patch Shapes - Noisy: {noisy_patch.shape}, Clean: {clean_patch.shape}"
        )

        # Assertions
        assert noisy_patch.shape == (
            1,
            Config.PATCH_SIZE,
            Config.PATCH_SIZE,
        ), "Train noisy patch shape mismatch"
        assert clean_patch.shape == (
            1,
            Config.PATCH_SIZE,
            Config.PATCH_SIZE,
        ), "Train clean patch shape mismatch"
        assert torch.is_tensor(noisy_patch), "Output should be a torch Tensor"
        assert (
            noisy_patch.max() <= 1.0 and noisy_patch.min() >= 0.0
        ), "Pixel values should be normalized [0,1]"

    # Test Val Dataset (Full Image)
    val_ds = DenoisingDataset(mode="val", load_cached_data=True)
    print(f"Validation Dataset initialized. Total samples (images): {len(val_ds)}")

    if len(val_ds) > 0:
        noisy_img, clean_img, img_id = val_ds[0]
        print(
            f"Sample Val Image Shapes - Noisy: {noisy_img.shape}, Clean: {clean_img.shape}, ID: {img_id}"
        )
        assert len(noisy_img.shape) == 3, "Val image should be [C, H, W]"

    # 3. Model Verification
    print("\n[3] Verifying Model Architecture...")
    device = Config.DEVICE
    model = CoRes2NetUNet().to(device)

    # Create a dummy input matching patch size
    dummy_input = torch.randn(1, 1, Config.PATCH_SIZE, Config.PATCH_SIZE).to(device)

    # Forward pass check
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Forward Pass - Input: {dummy_input.shape}, Output: {output.shape}")
    assert (
        output.shape == dummy_input.shape
    ), "Model output shape must match input shape"

    # 4. Training Loop Demonstration
    print("\n[4] Executing Training Loop (Demo)...")
    # We run for 1 epoch with a small batch size to demonstrate the loop functions correctly
    # and produces a checkpoint.
    best_rmse = run_training(num_epochs=1, batch_size=8, learning_rate=1e-3, patience=1)

    print(f"Training finished. Best RMSE: {best_rmse}")
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), f"Checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}"

    # 5. Metric Logic Verification
    print("\n[5] Verifying Metric Calculation...")
    # Test RMSE with known values
    # Pred: [0.0, 1.0], Target: [0.0, 0.0] -> Diff: [0, 1] -> Sq: [0, 1] -> Mean: 0.5 -> Sqrt: ~0.707
    p_test = np.array([0.0, 1.0])
    t_test = np.array([0.0, 0.0])
    calculated_rmse = calculate_rmse(p_test, t_test)
    expected_rmse = np.sqrt(0.5)

    print(
        f"Metric Check - Calculated: {calculated_rmse:.4f}, Expected: {expected_rmse:.4f}"
    )
    assert np.isclose(
        calculated_rmse, expected_rmse
    ), "RMSE calculation logic is incorrect"

    # 6. Inference and Submission
    print("\n[6] Generating Submission...")
    # This uses the checkpoint generated in step 4
    # We must pass the updated checkpoint path explicitly because the default arg was evaluated at import time
    create_submission(checkpoint_path=Config.MODEL_CHECKPOINT_PATH)

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created at: {Config.SUBMISSION_PATH}")
        print(f"Number of rows: {len(df_sub)}")
        print("First 5 rows:")
        print(df_sub.head())

        assert list(df_sub.columns) == ["id", "value"], "Submission columns mismatch"
        assert len(df_sub) > 0, "Submission file is empty"
        # Check value range
        assert (
            df_sub["value"].min() >= 0 and df_sub["value"].max() <= 1
        ), "Values out of range [0,1]"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
