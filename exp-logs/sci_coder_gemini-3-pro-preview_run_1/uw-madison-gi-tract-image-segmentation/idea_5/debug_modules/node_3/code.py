import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import (
    set_seed,
    get_dice_coef,
    get_3d_hausdorff,
    rle_decode,
    rle_encode,
)
from library.dataset import prepare_loaders
from library.model import UnetPlusPlus
from library.loss import WeightedDeepSupervisionLoss
from library.train import run_training
from library.inference import predict_and_submit


def main():
    # 1. Setup and Configuration
    print("--- 1. Setup and Configuration ---")
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    set_seed(42)

    # Modify Config for a fast demo run
    print("Modifying Config for speed...")
    Config.set_debug_mode(True)  # Subsets data, reduces epochs/batch_size
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.IMAGE_SIZE = (160, 160)  # Smaller resolution for speed (divisible by 32)
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory is clean for this run if needed,
    # but Config handles creation.
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading & Verification
    print("\n--- 2. Data Loading & Verification ---")
    # We use load_cached_data=False to ensure we process the subset correctly in debug mode
    train_loader, val_loader = prepare_loaders(load_cached_data=False, debug=True)

    # Fetch one batch
    images, masks = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Mask Shape: {masks.shape}")

    # Assertions
    # Shape: (Batch, Channels, Height, Width)
    # Channels should be 4 (3 RGB + 1 Depth)
    assert images.shape == (
        Config.BATCH_SIZE,
        4,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    ), f"Incorrect image shape: {images.shape}"

    # Masks: (Batch, Classes, Height, Width)
    assert masks.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    ), f"Incorrect mask shape: {masks.shape}"

    # Check Normalization (0-1 range)
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Images not normalized to [0, 1]"

    # Check Depth Channel (4th channel)
    # In a batch, depth might vary, but within one image (H, W) it should be constant
    depth_channel = images[0, 3, :, :]
    assert torch.all(
        depth_channel == depth_channel[0, 0]
    ), "Depth channel is not constant per image"

    print("Data loading verified successfully.")

    # 3. Model & Loss Logic Verification
    print("\n--- 3. Model & Loss Logic Verification ---")
    device = Config.DEVICE
    model = UnetPlusPlus().to(device)
    criterion = WeightedDeepSupervisionLoss()

    images = images.to(device)
    masks = masks.to(device)

    # Forward Pass
    model.train()  # Enable Deep Supervision
    outputs = model(images)

    # Check Deep Supervision Output
    assert isinstance(
        outputs, list
    ), "Model should return a list in training mode (Deep Supervision)"
    assert len(outputs) == 4, f"Expected 4 outputs (Final + 3 Aux), got {len(outputs)}"
    assert (
        outputs[0].shape == masks.shape
    ), f"Output shape mismatch. Got {outputs[0].shape}"

    # Loss Calculation
    loss = criterion(outputs, masks)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("Model and Loss logic verified.")

    # 4. Training Loop Simulation
    print("\n--- 4. Running Training Loop ---")
    # This runs the Trainer.fit() method
    run_training()

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
    print(f"Checkpoint found at: {checkpoint_path}")

    # 5. Inference & Submission
    print("\n--- 5. Running Inference & Submission ---")
    # This runs the inference pipeline
    predict_and_submit()

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify Submission Content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission rows: {len(sub_df)}")
    print("Submission head:")
    print(sub_df.head())

    required_cols = {"id", "class", "predicted"}
    assert required_cols.issubset(
        sub_df.columns
    ), f"Missing columns in submission. Found: {sub_df.columns}"

    # Verify RLE format on a dummy string if submission is empty (likely in debug with random weights)
    # or check a real one if present.
    if len(sub_df) > 0:
        rle_sample = sub_df.iloc[0]["predicted"]
        # It might be NaN/empty if model predicted nothing
        if pd.notna(rle_sample) and rle_sample != "":
            # Try decoding
            # We don't know the exact shape here easily without metadata,
            # but rle_decode just needs a shape that fits the indices.
            # We'll just check if it's a string of numbers.
            parts = rle_sample.split()
            assert all(p.isdigit() for p in parts), "RLE contains non-digit characters"
            assert len(parts) % 2 == 0, "RLE must have pairs of (start, length)"

    print("Inference pipeline verified.")

    # 6. Metric Utilities Verification
    print("\n--- 6. Metric Utilities Verification ---")
    # Create synthetic ground truth and prediction
    # Shape: (Depth, Height, Width)
    shape = (5, 100, 100)
    y_true = np.zeros(shape, dtype=np.uint8)
    y_pred = np.zeros(shape, dtype=np.uint8)

    # Add a square object
    y_true[2, 20:40, 20:40] = 1
    y_pred[2, 20:40, 20:40] = 1  # Perfect overlap

    dice = get_dice_coef(y_true, y_pred)
    hd = get_3d_hausdorff(y_true, y_pred)

    print(f"Perfect Match -> Dice: {dice:.4f}, HD: {hd:.4f}")
    assert np.isclose(dice, 1.0), "Dice should be 1.0 for perfect match"
    assert np.isclose(hd, 0.0), "Hausdorff should be 0.0 for perfect match"

    # Offset prediction to lower Dice and increase HD
    y_pred_bad = np.zeros(shape, dtype=np.uint8)
    y_pred_bad[2, 25:45, 25:45] = 1  # Shifted by 5 pixels

    dice_bad = get_dice_coef(y_true, y_pred_bad)
    hd_bad = get_3d_hausdorff(y_true, y_pred_bad)

    print(f"Offset Match -> Dice: {dice_bad:.4f}, HD: {hd_bad:.4f}")
    assert dice_bad < 1.0, "Dice should be < 1.0 for imperfect match"
    assert hd_bad > 0.0, "Hausdorff should be > 0.0 for imperfect match"

    print("Metric utilities verified.")
    print("\n=== All Tasks Completed Successfully ===")


if __name__ == "__main__":
    main()
