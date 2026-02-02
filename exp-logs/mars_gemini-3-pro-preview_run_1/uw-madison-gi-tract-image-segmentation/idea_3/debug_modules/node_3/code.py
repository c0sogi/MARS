import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.config import Config
from library.utils import (
    set_seed,
    rle_encode,
    rle_decode,
    compute_dice_coefficient,
    compute_hausdorff_distance,
)
from library.data import prepare_loaders
from library.model import UNet25D
from library.loss import BCEDiceLoss
from library.trainer import Trainer


def run_demo():
    print("=== Starting Demonstration Script ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid execution...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for demo
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1  # Single epoch to prove the loop works
    Config.NUM_WORKERS = 2  # Reduce overhead for small data

    # Redirect outputs to a demo folder in working directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.PREDICTION_DIR = os.path.join(Config.WORKING_DIR, "predictions")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    Config.setup()

    # Set reproducible seed
    set_seed(Config.SEED)
    print("Configuration complete.")

    # ---------------------------------------------------------
    # 2. Verify Utilities (RLE & Metrics)
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test RLE Encode/Decode Roundtrip
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1  # Create a square

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, (100, 100))

    if not np.array_equal(dummy_mask, decoded):
        raise AssertionError("RLE Encode/Decode roundtrip failed!")
    print("RLE Encode/Decode logic verified.")

    # Test Dice Coefficient
    # Case 1: Perfect overlap
    dice_perfect = compute_dice_coefficient(dummy_mask, dummy_mask)
    if not np.isclose(dice_perfect, 1.0):
        raise AssertionError(
            f"Dice score for perfect overlap should be 1.0, got {dice_perfect}"
        )

    # Case 2: No overlap
    empty_mask = np.zeros_like(dummy_mask)
    dice_empty = compute_dice_coefficient(dummy_mask, empty_mask)
    if not np.isclose(dice_empty, 0.0):
        raise AssertionError(
            f"Dice score for no overlap should be 0.0, got {dice_empty}"
        )
    print("Dice metric logic verified.")

    # Test Hausdorff (3D)
    # Create simple 3D volumes (Depth, Height, Width)
    vol_true = np.zeros((5, 100, 100), dtype=np.uint8)
    vol_pred = np.zeros((5, 100, 100), dtype=np.uint8)

    vol_true[2, 50, 50] = 1
    vol_pred[2, 50, 50] = 1
    hd_perfect = compute_hausdorff_distance(vol_true, vol_pred)

    # Note: HD is distance, so perfect match is 0.0
    if not np.isclose(hd_perfect, 0.0):
        raise AssertionError(
            f"Hausdorff distance for perfect match should be 0.0, got {hd_perfect}"
        )

    print("Hausdorff metric logic verified.")

    # ---------------------------------------------------------
    # 3. Verify Data Loading
    # ---------------------------------------------------------
    print("\n[3] Verifying Data Loading...")

    train_loader, val_loader, test_loader = prepare_loaders(debug=True)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    masks = batch["mask"]
    ids = batch["id"]

    # Check Shapes
    # Image: (B, 3, H, W) -> 3 channels because of 2.5D input (z-1, z, z+1)
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    if images.shape != expected_img_shape:
        raise AssertionError(
            f"Image batch shape mismatch. Expected {expected_img_shape}, got {images.shape}"
        )

    # Mask: (B, 3, H, W) -> 3 classes
    expected_mask_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    if masks.shape != expected_mask_shape:
        raise AssertionError(
            f"Mask batch shape mismatch. Expected {expected_mask_shape}, got {masks.shape}"
        )

    # Check Data Types and Ranges
    if images.dtype != torch.float32:
        raise AssertionError("Images should be float32")
    if (
        masks.dtype != torch.float32
    ):  # BCEWithLogits expects float targets usually, or we cast later
        raise AssertionError("Masks should be float32")

    if images.max() > 1.0 or images.min() < 0.0:
        print(
            f"Warning: Image values out of [0,1] range. Max: {images.max()}, Min: {images.min()}"
        )
        # Note: Depending on normalization strategy, this might be expected, but Utils.load_slice_img does 0-1 scaling.
        # We assert strictly here based on code analysis.
        if images.max() > 1.0 + 1e-5 or images.min() < 0.0 - 1e-5:
            raise AssertionError("Image normalization failed. Values outside [0, 1].")

    print("Data Loader shapes and types verified.")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = UNet25D(
        backbone_name=Config.BACKBONE, pretrained=False
    )  # No need to download weights for shape check
    model.to(Config.DEVICE)
    model.eval()

    with torch.no_grad():
        # Move batch to device
        imgs_gpu = images.to(Config.DEVICE)
        output = model(imgs_gpu)

    if output.shape != expected_mask_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_mask_shape}, got {output.shape}"
        )

    print("Model forward pass verified.")

    # Verify Loss Function
    criterion = BCEDiceLoss()
    masks_gpu = masks.to(Config.DEVICE)
    loss = criterion(output, masks_gpu)

    if not torch.isfinite(loss):
        raise AssertionError("Loss is NaN or Infinite.")

    print(f"Loss calculation verified. Initial Loss: {loss.item():.4f}")

    # ---------------------------------------------------------
    # 5. Verify Training Loop (Trainer)
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch, Debug Data)...")

    # Re-initialize trainer with loaded data
    trainer = Trainer(train_loader, val_loader, test_loader)

    # Run Fit
    # This runs training, validation, and prediction steps
    trainer.fit()

    # Check Artifacts
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise AssertionError("Training finished but 'best_model.pth' was not saved.")

    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise AssertionError(
            "Training finished but 'submission.csv' was not generated."
        )

    # Verify Submission Content
    df_sub = pd.read_csv(submission_path)
    required_cols = {"id", "class", "predicted"}
    if not required_cols.issubset(df_sub.columns):
        raise AssertionError(
            f"Submission file missing columns. Found: {df_sub.columns}"
        )

    if len(df_sub) == 0:
        print(
            "Warning: Submission file is empty. This might be expected if test set is empty in debug mode, but usually test metadata has rows."
        )
    else:
        print(f"Submission generated with {len(df_sub)} rows.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
