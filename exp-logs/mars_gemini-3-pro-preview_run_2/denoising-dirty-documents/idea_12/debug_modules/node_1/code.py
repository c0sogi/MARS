import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.model import CoSPResUNet
from library.dataset import TextDenoisingDataset
from library.train import train_model
from library.predict import predict_tiled, predict_with_tta, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # --- 1. Configuration Overrides for Fast Execution ---
    # We modify the Config class attributes directly to set up a demo environment
    # that runs quickly and doesn't interfere with main experiments.
    print("\n[1] Configuring environment for demo...")

    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_test.csv")

    # Reduce training intensity
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.PATCHES_PER_EPOCH = 2  # Extract fewer patches per image per epoch
    Config.NUM_WORKERS = 2  # Reduce workers for simple demo

    # Initialize directories and seeds
    Config.initialize()
    seed_everything(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print("Configuration updated successfully.")

    # --- 2. Model Logic Verification ---
    print("\n[2] Verifying Model Architecture...")
    device = Config.DEVICE
    model = CoSPResUNet().to(device)

    # Create a dummy input tensor: (Batch=2, Channels=1, Height=128, Width=128)
    dummy_input = torch.randn(2, 1, 128, 128).to(device)

    # Forward pass
    try:
        output = model(dummy_input)
    except Exception as e:
        raise RuntimeError(f"Model forward pass failed: {e}")

    # Assertions
    # The model is a U-Net, so input shape should equal output shape
    assert (
        output.shape == dummy_input.shape
    ), f"Shape mismatch! Input: {dummy_input.shape}, Output: {output.shape}"

    # Check if output is a tensor and requires grad (if training)
    assert isinstance(output, torch.Tensor), "Output is not a Tensor"

    print("Model architecture verified. Input/Output shapes match.")

    # --- 3. Dataset Logic Verification ---
    print("\n[3] Verifying Dataset Loading...")

    # Initialize dataset in train mode
    train_ds = TextDenoisingDataset(
        metadata_path=Config.TRAIN_METADATA,
        mode="train",
        # We rely on the internal get_transforms logic via the class or pass None to test raw loading
        # Here we let it use the default behavior if transform is None, but the class expects one usually.
        # Let's import get_transforms to be safe.
    )
    from library.dataset import get_transforms

    train_ds.transform = get_transforms("train")

    # Check length
    # Length = num_images * patches_per_epoch
    # We have 92 train images in metadata (from description) and set patches_per_epoch=2
    expected_len = 92 * Config.PATCHES_PER_EPOCH
    assert (
        len(train_ds) == expected_len
    ), f"Dataset length mismatch. Expected {expected_len}, got {len(train_ds)}"

    # Fetch one sample
    noisy, clean = train_ds[0]

    # Assertions
    assert isinstance(noisy, torch.Tensor), "Noisy image is not a Tensor"
    assert isinstance(clean, torch.Tensor), "Clean image is not a Tensor"
    assert noisy.shape == (1, 128, 128), f"Unexpected noisy patch shape: {noisy.shape}"
    assert clean.shape == (1, 128, 128), f"Unexpected clean patch shape: {clean.shape}"

    print("Dataset loading verified. Shapes and types are correct.")

    # --- 4. Training Loop Demonstration ---
    print("\n[4] Running Short Training Loop...")

    # We use a very small max_train_samples to make this finish in seconds
    # This tests the integration of DataLoader, Model, Loss, and Optimizer
    train_model(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        max_train_samples=16,  # Only use 16 patches for training demo
    )

    # Verify model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"

    print("Training loop completed successfully. Checkpoint saved.")

    # --- 5. Inference Logic Verification ---
    print("\n[5] Verifying Inference Logic...")

    # Load the trained model
    checkpoint = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create a dummy full-sized image (e.g., 540x420 like the dataset)
    # Shape: (1, 1, H, W)
    dummy_full_img = torch.rand(1, 1, 420, 540).to(device)

    # Test Tiled Prediction
    tiled_output = predict_tiled(
        model, dummy_full_img, patch_size=128, overlap=0.5, device=device
    )

    # Assertions for Tiled Prediction
    assert (
        tiled_output.shape == dummy_full_img.shape
    ), f"Tiled output shape mismatch. Expected {dummy_full_img.shape}, got {tiled_output.shape}"

    # Test TTA Prediction (Test Time Augmentation)
    tta_output = predict_with_tta(
        model, dummy_full_img, patch_size=128, overlap=0.5, device=device
    )

    # TTA returns a numpy array of shape (H, W)
    assert isinstance(tta_output, np.ndarray), "TTA output should be numpy array"
    assert tta_output.shape == (
        420,
        540,
    ), f"TTA output shape mismatch. Expected (420, 540), got {tta_output.shape}"

    print("Inference functions (Tiled & TTA) verified.")

    # --- 6. Submission Generation Verification ---
    print("\n[6] Generating Submission File...")

    # This function iterates over the test set defined in metadata/test.csv
    # The test set is small (29 images), so we can run it fully.
    generate_submission(
        model_path=Config.MODEL_SAVE_PATH,
        output_path=Config.SUBMISSION_PATH,
        batch_size=1,
        device=device,
    )

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission file missing required columns 'id' or 'value'"

    # Check content format (id should be string, value should be numeric)
    assert len(df_sub) > 0, "Submission file is empty"
    assert isinstance(df_sub.iloc[0]["id"], str), "ID column should be string"

    # Check ID format (image_row_col)
    example_id = df_sub.iloc[0]["id"]
    parts = example_id.split("_")
    assert len(parts) >= 3, f"ID format incorrect: {example_id}"

    print(f"Submission generated successfully with {len(df_sub)} rows.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
