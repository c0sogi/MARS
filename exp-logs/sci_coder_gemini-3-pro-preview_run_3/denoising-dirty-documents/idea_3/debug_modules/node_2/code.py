import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import (
    set_seed,
    calculate_rmse,
    normalize_image,
    denormalize_image,
    get_cached_data,
)
from library.model import DRDN
from library.dataset import get_dataloaders
from library.trainer import Trainer
from library.inference import generate_predictions

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting DRDN Library Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config parameters for speed
    # We use a separate working directory for the demo to avoid overwriting actual run artifacts
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Reduce computational load
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.PATCH_SIZE = 64
    # Large stride to generate very few patches for speed
    Config.PATCH_STRIDE = 200
    Config.EARLY_STOPPING_PATIENCE = 2

    # Ensure directories exist
    Config.create_directories()

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Model Path: {Config.MODEL_SAVE_PATH}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying utility functions...")

    # Test Normalization
    dummy_img_uint8 = np.array([[0, 127, 255]], dtype=np.uint8)
    norm_img = normalize_image(dummy_img_uint8)
    assert norm_img.dtype == np.float32, "Normalized image should be float32"
    assert np.isclose(norm_img.min(), 0.0), "Min value should be 0.0"
    assert np.isclose(norm_img.max(), 1.0), "Max value should be 1.0"

    # Test Denormalization
    denorm_img = denormalize_image(norm_img)
    assert denorm_img.dtype == np.uint8, "Denormalized image should be uint8"
    assert denorm_img[0, 0] == 0
    assert denorm_img[0, 2] == 255

    # Test RMSE
    pred = np.array([0.5, 0.5])
    target = np.array(
        [0.5, 0.9]
    )  # Diff is 0, 0.4. Sq: 0, 0.16. Mean: 0.08. Sqrt(0.08) ~= 0.2828
    rmse = calculate_rmse(pred, target)
    expected_rmse = np.sqrt(0.08)
    assert np.isclose(
        rmse, expected_rmse
    ), f"RMSE calculation incorrect. Got {rmse}, expected {expected_rmse}"

    print("Utility functions verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Data Loading...")

    # We use a debug_limit to load only a tiny subset of patches
    # Note: The first run will process images and cache the result.
    # Subsequent runs would load from cache.
    train_loader, val_loader = get_dataloaders(
        load_cached_data=False,  # Force re-compute for demo purposes to ensure logic runs
        batch_size=Config.BATCH_SIZE,
        debug_limit=20,  # Only use 20 patches
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")

    # Verify batch structure
    for inputs, targets in train_loader:
        assert inputs.dim() == 4, "Input should be 4D tensor (B, C, H, W)"
        assert targets.dim() == 4, "Target should be 4D tensor (B, C, H, W)"
        assert inputs.shape[1] == 1, "Input should have 1 channel"
        assert inputs.shape[2] == Config.PATCH_SIZE, "Height mismatch"
        assert inputs.shape[3] == Config.PATCH_SIZE, "Width mismatch"

        # Check value range
        assert inputs.max() <= 1.0 and inputs.min() >= 0.0, "Inputs not normalized"
        break  # Check only first batch

    print("DataLoaders functioning correctly.")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = DRDN().to(device)

    # Create dummy input: (Batch=2, Channel=1, H=64, W=64)
    dummy_input = torch.randn(2, 1, 64, 64).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    assert (
        output.shape == dummy_input.shape
    ), "Output shape must match input shape (residual learning)"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (Short Run)...")

    trainer = Trainer()

    # Override trainer parameters locally if needed, but Config override handles most
    # We call fit with debug_limit matching our dataloader test
    trainer.fit(load_cached_data=True, debug_limit=20)

    # Check if model checkpoint was saved
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Checkpoint successfully saved at {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created!")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference and Generating Submission...")

    # This function loads the model we just trained and predicts on the test set
    # The test set size is fixed by metadata, but inference is fast.
    generate_predictions()

    # Validate Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    # Check Header
    assert list(df_sub.columns) == ["id", "value"], "Submission columns mismatch"

    # Check ID format (e.g., "110_1_1")
    sample_id = df_sub.iloc[0]["id"]
    parts = sample_id.split("_")
    assert len(parts) >= 3, f"Invalid ID format: {sample_id}"

    # Check Value type (int64 as per sample submission description in prompt)
    # Note: The prompt mentioned sample submission has int64.
    # Our inference code produces uint8 (0-255), which pandas reads as int64.
    assert pd.api.types.is_integer_dtype(
        df_sub["value"]
    ), "Value column should be integer"
    assert df_sub["value"].min() >= 0, "Pixel values cannot be negative"
    assert df_sub["value"].max() <= 255, "Pixel values cannot exceed 255"

    print("Submission file format validated.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
