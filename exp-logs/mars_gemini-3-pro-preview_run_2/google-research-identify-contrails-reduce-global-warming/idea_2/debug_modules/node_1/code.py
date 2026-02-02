import os
import sys
import numpy as np
import torch
import pandas as pd
import warnings

# ==========================================
# 0. Setup & Configuration
# ==========================================

# Suppress tqdm progress bars to keep output clean as per requirements
# This must be done before importing modules that use tqdm
import tqdm


def no_op_tqdm(iterable=None, *args, **kwargs):
    if iterable is None:
        return
    return iterable


tqdm.tqdm = no_op_tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library modules
from library.config import DEVICE, MODEL_SAVE_PATH, SUBMISSION_FILE_PATH, WORKING_DIR
from library.utils import set_seed, rle_encode, dice_coef
from library.dataset import ContrailDataset, get_dataloader
from library.model import SymmetricUNetPlusPlus
from library.train import train_model
from library.predict import predict_and_submit


def main():
    print("Starting Contrail Identification Task Demonstration...")

    # Set seed for reproducibility
    set_seed(42)

    # ==========================================
    # 1. Verify Utility Functions
    # ==========================================
    print("\n[1/5] Verifying Utility Functions...")

    # Test RLE Encoding
    # Create a 3x3 mask:
    # 0 1 0
    # 0 1 0
    # 0 0 0
    # Flattened (Column-major/Fortran): 0, 0, 0, 1, 1, 0, 0, 0, 0
    # Runs: The 1s are at indices 4 and 5 (1-based). Run starts at 4, length 2.
    dummy_mask = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    encoded = rle_encode(dummy_mask)
    expected_rle = "4 2"
    assert (
        encoded == expected_rle
    ), f"RLE Encoding failed. Expected '{expected_rle}', got '{encoded}'"
    print("  RLE Encoding: OK")

    # Test Dice Coefficient
    # Pred: 1 1 0, True: 1 0 0
    # Intersection: 1, Union: 2 + 1 = 3. Dice = 2*1 / 3 = 0.666...
    y_pred = torch.tensor([1.0, 1.0, 0.0])
    y_true = torch.tensor([1.0, 0.0, 0.0])
    dice = dice_coef(y_pred, y_true).item()
    assert abs(dice - 0.666666) < 1e-4, f"Dice Coefficient failed. Got {dice}"
    print("  Dice Coefficient: OK")

    # ==========================================
    # 2. Verify Dataset & DataLoader
    # ==========================================
    print("\n[2/5] Verifying Dataset and DataLoader...")

    # Initialize Dataset in Debug mode (uses subset)
    # Split='train' includes masks
    ds = ContrailDataset(split="train", debug=True)
    print(f"  Dataset initialized with {len(ds)} samples (Debug Mode).")

    # Fetch a single sample
    img, mask = ds[0]

    # Check Shapes
    # Image: (C, H, W) where C=9 (3 current + 3 diff_prev + 3 diff_next)
    # Mask: (C, H, W) where C=1
    assert img.shape == (9, 256, 256), f"Incorrect image shape: {img.shape}"
    assert mask.shape == (1, 256, 256), f"Incorrect mask shape: {mask.shape}"
    assert isinstance(img, torch.Tensor), "Image is not a Tensor"
    print("  Sample shapes: OK")

    # Check DataLoader
    batch_size = 4
    loader = get_dataloader("train", batch_size=batch_size, debug=True)
    batch_imgs, batch_masks = next(iter(loader))

    assert batch_imgs.shape == (
        batch_size,
        9,
        256,
        256,
    ), f"Batch image shape mismatch: {batch_imgs.shape}"
    assert batch_masks.shape == (
        batch_size,
        1,
        256,
        256,
    ), f"Batch mask shape mismatch: {batch_masks.shape}"
    print("  DataLoader batching: OK")

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n[3/5] Verifying Model Architecture...")

    model = SymmetricUNetPlusPlus(in_channels=9, n_classes=1).to(DEVICE)

    # Perform Forward Pass with the batch loaded earlier
    # Move batch to device
    batch_imgs = batch_imgs.to(DEVICE)

    with torch.no_grad():
        logits = model(batch_imgs)

    # Output should be (Batch, 1, 256, 256)
    assert logits.shape == (
        batch_size,
        1,
        256,
        256,
    ), f"Model output shape mismatch: {logits.shape}"
    print("  Model forward pass: OK")

    # ==========================================
    # 4. Run Training Loop (Demonstration)
    # ==========================================
    print("\n[4/5] Running Training Loop (1 Epoch, Debug Mode)...")

    # Train for 1 epoch to generate the model file
    # We use the default save path defined in config.py
    train_model(debug=True, epochs=1, batch_size=8, lr=1e-4, patience=1)

    # Verify model file was created
    assert os.path.exists(MODEL_SAVE_PATH), f"Model file not found at {MODEL_SAVE_PATH}"
    print(f"  Training complete. Model saved to {MODEL_SAVE_PATH}")

    # ==========================================
    # 5. Run Inference & Submission
    # ==========================================
    print("\n[5/5] Running Inference and Generating Submission...")

    # Run prediction (uses the model saved in step 4)
    predict_and_submit(debug=True)

    # Verify submission file
    assert os.path.exists(
        SUBMISSION_FILE_PATH
    ), f"Submission file not found at {SUBMISSION_FILE_PATH}"

    df_sub = pd.read_csv(SUBMISSION_FILE_PATH)

    # Check Columns
    expected_cols = ["record_id", "encoded_pixels"]
    assert all(
        col in df_sub.columns for col in expected_cols
    ), "Submission columns mismatch"

    # Check Content
    # In debug mode, we process a subset. Ensure it's not empty.
    assert len(df_sub) > 0, "Submission file is empty"

    # Check RLE format (basic check)
    # Should be string or NaN (if empty, but pandas might load '-' as string)
    # The utils.py rle_encode returns '-' for empty, so it should be a string.
    sample_rle = df_sub.iloc[0]["encoded_pixels"]
    assert isinstance(sample_rle, str), "Encoded pixels should be string"

    print(f"  Submission generated successfully with {len(df_sub)} records.")
    print(f"  Sample: {df_sub.iloc[0].to_dict()}")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
