import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import (
    set_seed,
    rle_encode,
    rle_decode,
    get_dice_score,
    get_3d_hausdorff,
)
from library.dataset import process_metadata, UWMadissonDataset, get_transforms
from library.model import SegmentationModel
from library.loss import BCETverskyLoss
from library.trainer import Trainer


def run_demo():
    print("=== Starting Demonstration ===")

    # 1. Setup and Configuration Overrides for Speed
    set_seed(Config.SEED)

    # Override Config for a quick demo run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    # Ensure output directories exist (Config creates them on import, but good to be safe)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Device: {Config.DEVICE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # 2. Data Loading and Preparation
    print("\n--- Preparing Data ---")
    # Load metadata
    # We use the pre-generated metadata file
    full_df = process_metadata(Config.TRAIN_CSV, mode="train", load_cached_data=False)

    # Subset data for speed: Select only 2 cases
    # One for training, one for validation to simulate a real split
    unique_cases = full_df["case"].unique()
    if len(unique_cases) < 2:
        raise ValueError("Not enough cases in metadata for demo split.")

    train_case = unique_cases[0]
    val_case = unique_cases[1]

    train_df = full_df[full_df["case"] == train_case].reset_index(drop=True)
    val_df = full_df[full_df["case"] == val_case].reset_index(drop=True)

    # Further limit slices to ensure very fast epochs (e.g., first 20 slices)
    train_df = train_df.head(20)
    val_df = val_df.head(20)

    print(f"Train subset size: {len(train_df)} (Case: {train_case})")
    print(f"Val subset size: {len(val_df)} (Case: {val_case})")

    # Create Datasets
    train_dataset = UWMadissonDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = UWMadissonDataset(
        val_df, transforms=get_transforms("valid"), mode="valid"
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Verify Model and Loss
    print("\n--- Verifying Model and Loss ---")
    model = SegmentationModel().to(Config.DEVICE)
    criterion = BCETverskyLoss()

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["image"].to(Config.DEVICE)
    masks = batch["mask"].to(Config.DEVICE)

    print(f"Input Image Shape: {images.shape}")
    print(f"Target Mask Shape: {masks.shape}")

    # Forward pass
    with torch.amp.autocast("cuda", enabled=Config.MIXED_PRECISION):
        logits = model(images)
        loss = criterion(logits, masks)

    print(f"Output Logits Shape: {logits.shape}")
    print(f"Calculated Loss: {loss.item():.4f}")

    # Assertions
    assert logits.shape == masks.shape, "Model output shape mismatch!"
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be positive"

    # 4. Run Trainer (Training Loop)
    print("\n--- Running Trainer (1 Epoch) ---")
    trainer = Trainer(train_loader, val_loader)

    # We already instantiated a model in the trainer, but let's ensure settings are correct
    # The Trainer class handles the loop, validation, and metric calculation
    trainer.fit()

    # Check if model files were created
    assert os.path.exists(Config.LAST_MODEL_PATH), "Last model checkpoint not found!"
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model checkpoint not found!"
    print("Training loop completed successfully.")

    # 5. Verify Utilities (RLE & Metrics)
    print("\n--- Verifying Utilities ---")

    # Test RLE Encoding/Decoding
    # Create a simple 10x10 mask with a 2x2 square of 1s at (1,1)
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    dummy_mask[1:3, 1:3] = 1

    rle_str = rle_encode(dummy_mask)
    decoded_mask = rle_decode(rle_str, shape=(10, 10))

    assert np.array_equal(
        dummy_mask, decoded_mask
    ), "RLE Decode does not match original mask!"
    print("RLE Encode/Decode: Passed")

    # Test Dice Score
    # Perfect overlap
    dice_perfect = get_dice_score(dummy_mask, dummy_mask)
    assert np.isclose(
        dice_perfect, 1.0
    ), f"Dice score for perfect match should be 1.0, got {dice_perfect}"

    # No overlap
    inverse_mask = 1 - dummy_mask
    dice_zero = get_dice_score(dummy_mask, inverse_mask)
    assert np.isclose(
        dice_zero, 0.0, atol=1e-4
    ), f"Dice score for no overlap should be 0.0, got {dice_zero}"
    print("Dice Score: Passed")

    # Test 3D Hausdorff
    # Create two small 3D volumes (D=2, H=10, W=10)
    vol_pred = np.zeros((2, 10, 10), dtype=np.uint8)
    vol_true = np.zeros((2, 10, 10), dtype=np.uint8)

    # Slice 0: identical square
    vol_pred[0, 1:3, 1:3] = 1
    vol_true[0, 1:3, 1:3] = 1

    # Slice 1: pred is shifted by 1 pixel compared to true
    vol_pred[1, 5, 5] = 1
    vol_true[1, 5, 6] = 1

    hd = get_3d_hausdorff(vol_pred, vol_true)
    print(f"3D Hausdorff Distance: {hd:.4f}")

    # Distance should be non-zero because of the shift in slice 1
    # Normalized coords: 1 pixel shift in W=10 is 0.1 distance
    assert hd > 0.0, "Hausdorff distance should be > 0 for mismatched volumes"
    assert hd < 1.0, "Hausdorff distance should be < 1 for close volumes"
    print("3D Hausdorff: Passed")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
