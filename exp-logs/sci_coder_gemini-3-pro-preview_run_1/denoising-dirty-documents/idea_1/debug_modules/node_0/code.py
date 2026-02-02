import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, pad_image, unpad_image, calculate_rmse
from library.dataset import DenoisingDataset, get_dataloaders
from library.model import UNet
from library.train import train_model
from library.predict import generate_submission as predict_submission


def main():
    print("=== Denoising Library Usage Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration...")
    # Initialize the environment
    Config.setup()

    # Override configuration for a quick demonstration
    # We use a separate cache directory to avoid overwriting existing work
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.CACHE_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Set hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    print(f"    Cache Directory: {Config.CACHE_DIR}")
    print(f"    Epochs: {Config.NUM_EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test: pad_image and unpad_image
    # Create a tensor with dimensions NOT divisible by 32 (e.g., 100x100)
    h_orig, w_orig = 100, 100
    dummy_img = torch.randn(1, 1, h_orig, w_orig)

    # Pad
    padded_img = pad_image(dummy_img, divisor=32, mode="reflect")
    h_pad, w_pad = padded_img.shape[2], padded_img.shape[3]

    # Assert dimensions are divisible by 32
    assert (
        h_pad % 32 == 0 and w_pad % 32 == 0
    ), "Padding failed to produce dimensions divisible by 32."
    assert h_pad >= h_orig and w_pad >= w_orig, "Padded image is smaller than original."

    # Unpad
    unpadded_img = unpad_image(padded_img, (h_orig, w_orig))

    # Assert dimensions returned to original
    assert (
        unpadded_img.shape == dummy_img.shape
    ), "Unpadding failed to restore original dimensions."
    assert torch.allclose(
        dummy_img, unpadded_img
    ), "Unpadded content differs from original."

    # Test: calculate_rmse
    t1 = torch.tensor([1.0, 2.0, 3.0])
    t2 = torch.tensor([1.0, 2.0, 3.0])
    rmse_val = calculate_rmse(t1, t2)
    assert rmse_val == 0.0, f"RMSE of identical tensors should be 0.0, got {rmse_val}"

    t3 = torch.tensor(
        [1.0, 2.0, 7.0]
    )  # Error of 4 on last element. MSE = 16/3. RMSE = sqrt(5.33) ~= 2.309
    rmse_val_err = calculate_rmse(t1, t3)
    assert rmse_val_err > 0, "RMSE should be positive for different tensors."

    print("    Utilities verified successfully.")

    # ---------------------------------------------------------
    # 3. Verify Dataset and DataLoaders
    # ---------------------------------------------------------
    print("\n[3] Verifying Dataset and DataLoaders...")

    # Use a small limit to speed up loading
    DATASET_LIMIT = 4

    # Instantiate Datasets
    train_ds = DenoisingDataset(
        Config.TRAIN_METADATA_PATH, mode="train", limit=DATASET_LIMIT
    )
    val_ds = DenoisingDataset(Config.VAL_METADATA_PATH, mode="val", limit=DATASET_LIMIT)
    test_ds = DenoisingDataset(
        Config.TEST_METADATA_PATH, mode="test", limit=DATASET_LIMIT
    )

    # Verify Train Item (Random Crop)
    noisy, clean = train_ds[0]
    # Shape should be (1, PATCH_SIZE, PATCH_SIZE)
    assert noisy.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Train noisy shape mismatch: {noisy.shape}"
    assert clean.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Train clean shape mismatch: {clean.shape}"

    # Verify Val Item (Full Image)
    noisy_val, clean_val = val_ds[0]
    # Shape should be (1, H, W)
    assert (
        noisy_val.dim() == 3 and noisy_val.shape[0] == 1
    ), "Val noisy tensor dimension incorrect."

    # Verify Test Item (Noisy + ID)
    noisy_test, img_id = test_ds[0]
    assert isinstance(img_id, str), "Test dataset ID should be a string."

    # Verify DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(dataset_limit=DATASET_LIMIT)

    # Fetch one batch from train loader
    batch_noisy, batch_clean = next(iter(train_loader))
    assert (
        batch_noisy.shape[0] == Config.BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {batch_noisy.shape[0]}"

    print("    Datasets and DataLoaders verified successfully.")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = UNet(n_channels=Config.NUM_CHANNELS, n_classes=1, bilinear=True).to(device)

    # Create dummy input: (Batch, Channel, Height, Width)
    # Using 128x128 as it's a standard patch size
    dummy_input = torch.randn(2, 1, 128, 128).to(device)

    # Forward pass
    output = model(dummy_input)

    # Check output shape
    assert (
        output.shape == dummy_input.shape
    ), f"Model output shape mismatch. Expected {dummy_input.shape}, got {output.shape}"

    print("    Model architecture verified successfully.")

    # ---------------------------------------------------------
    # 5. Demonstrate Training Loop
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop (Demo)...")

    # We use the provided high-level function `train_model`
    # It handles initialization, loop, validation, and saving.
    # We pass dataset_limit to ensure it runs quickly.

    train_model(epochs=1, batch_size=2, dataset_limit=DATASET_LIMIT)

    # Verify that the checkpoint was saved
    if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        print(f"    Checkpoint successfully created at: {Config.MODEL_CHECKPOINT_PATH}")
    else:
        raise AssertionError("Model checkpoint was not created after training.")

    # ---------------------------------------------------------
    # 6. Demonstrate Inference / Prediction
    # ---------------------------------------------------------
    print("\n[6] Running Inference (Demo)...")

    # We use `predict_submission` which loads the saved model and generates the CSV.
    # Note: train_model also generates a submission, but we run this explicitly
    # to demonstrate the standalone prediction capability.

    predict_submission(dataset_limit=DATASET_LIMIT)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"    Submission file successfully created at: {Config.SUBMISSION_PATH}")

        # Validate content format
        df = pd.read_csv(Config.SUBMISSION_PATH)
        assert list(df.columns) == ["id", "value"], "Submission columns are incorrect."
        assert not df.empty, "Submission dataframe is empty."

        # Check first row ID format
        first_id = df.iloc[0]["id"]
        # Expected format: {img_id}_{row}_{col}
        assert (
            len(str(first_id).split("_")) >= 3
        ), f"Submission ID format incorrect: {first_id}"

        print(f"    Generated {len(df)} pixel predictions.")
    else:
        raise AssertionError("Submission file was not created after inference.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
