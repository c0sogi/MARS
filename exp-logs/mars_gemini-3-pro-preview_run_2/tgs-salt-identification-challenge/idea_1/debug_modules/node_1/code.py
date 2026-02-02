import os
import torch
import numpy as np
import pandas as pd
import cv2
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, metric_iou
from library.dataset import SaltDataset
from library.model import DepthLinkNet
from library.train import Trainer
from library.predict import generate_submission


def main():
    print("Starting Task Demonstration...")

    # 1. Configuration Setup
    # We override defaults to ensure the demo runs quickly (1 epoch, small batch)
    config = Config(
        EPOCHS=1,
        BATCH_SIZE=4,
        WORKING_DIR="./working/demo_run",
        CHECKPOINT_PATH="./working/demo_run/best_model.pth",
        SUBMISSION_PATH="./working/demo_run/submission.csv",
        NUM_WORKERS=0,  # Use 0 workers to avoid multiprocessing overhead in a short script
    )

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(config.SEED)
    print("Configuration initialized and seed set.")

    # 2. Verify Utilities (RLE and IoU)
    print("\n--- Verifying Utilities ---")

    # Test RLE Encode/Decode
    # Create a simple 3x3 mask:
    # [[0, 1, 0],
    #  [0, 1, 0],
    #  [0, 1, 0]]
    # Flattened (Fortran/Column-major): 0, 0, 0, 1, 1, 1, 0, 0, 0 -> indices 4, 5, 6 (1-based)
    # Wait, Fortran flatten of the above:
    # Col 0: 0, 0, 0
    # Col 1: 1, 1, 1
    # Col 2: 0, 0, 0
    # Sequence: 0,0,0, 1,1,1, 0,0,0.
    # Runs: Start at index 4 (1-based), length 3.

    dummy_mask = np.zeros((3, 3), dtype=np.uint8)
    dummy_mask[:, 1] = 1

    rle_str = rle_encode(dummy_mask)
    print(f"RLE String for vertical line in 3x3: '{rle_str}'")

    # Decode back
    decoded_mask = rle_decode(rle_str, shape=(3, 3))

    if not np.array_equal(dummy_mask, decoded_mask):
        raise AssertionError("RLE Decode does not match original mask.")
    print("RLE Encode/Decode logic verified.")

    # Test IoU Metric
    # Case 1: Perfect match
    iou_perfect = metric_iou(dummy_mask, dummy_mask)
    if not np.isclose(iou_perfect, 1.0):
        raise AssertionError(f"IoU for perfect match should be 1.0, got {iou_perfect}")

    # Case 2: No overlap
    dummy_mask_inv = 1 - dummy_mask
    iou_zero = metric_iou(dummy_mask, dummy_mask_inv)
    if not np.isclose(iou_zero, 0.0):
        raise AssertionError(f"IoU for disjoint masks should be 0.0, got {iou_zero}")
    print("IoU Metric logic verified.")

    # 3. Verify Dataset
    print("\n--- Verifying Dataset ---")
    # Initialize dataset in train mode
    ds = SaltDataset(config.TRAIN_CSV, config, mode="train", load_cached_data=False)

    # Fetch one sample
    image, depth, mask = ds[0]

    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Depth Shape: {depth.shape}")
    print(f"Sample Mask Shape: {mask.shape}")

    # Check Shapes
    # Image: (1, 128, 128) -> Channel, Height, Width
    if image.shape != (1, 128, 128):
        raise AssertionError(f"Expected image shape (1, 128, 128), got {image.shape}")
    if mask.shape != (1, 128, 128):
        raise AssertionError(f"Expected mask shape (1, 128, 128), got {mask.shape}")
    if depth.shape != (1,):
        raise AssertionError(f"Expected depth shape (1,), got {depth.shape}")

    # Check Value Ranges
    if image.min() < 0 or image.max() > 1:
        raise AssertionError("Image data not normalized to [0, 1].")
    unique_mask_vals = torch.unique(mask)
    if not all(val in [0, 1] for val in unique_mask_vals):
        raise AssertionError("Mask is not binary (0 or 1).")

    print("Dataset shapes and value ranges verified.")

    # 4. Verify Model Architecture
    print("\n--- Verifying Model ---")
    model = DepthLinkNet(in_channels=config.CHANNELS, num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)
    model.eval()

    # Create dummy batch
    # Batch size 2, 1 channel, 128x128
    dummy_img = torch.randn(2, 1, 128, 128).to(config.DEVICE)
    dummy_depth = torch.randn(2, 1).to(config.DEVICE)

    with torch.no_grad():
        output = model(dummy_img, dummy_depth)

    print(f"Model Output Shape: {output.shape}")

    if output.shape != (2, 1, 128, 128):
        raise AssertionError(
            f"Expected output shape (2, 1, 128, 128), got {output.shape}"
        )
    print("Model forward pass successful.")

    # 5. Verify Training Loop (Short Run)
    print("\n--- Verifying Training Loop ---")
    trainer = Trainer(config)

    # Run training for 1 epoch on a tiny subset (32 samples)
    # This ensures the code runs quickly but exercises the full training path
    trainer.train(epochs=1, debug_limit=32)

    # Check if checkpoint was created
    if not os.path.exists(config.CHECKPOINT_PATH):
        # Note: Checkpoint is only saved if validation IoU improves.
        # With random init and 32 samples, it might not improve over 0.0 if the model predicts all 0s and masks are empty.
        # However, usually IoU starts at 0, so any correct pixel helps.
        # If it fails, we force save for the next step.
        print(
            "Checkpoint not created (likely no improvement). Saving manually for prediction demo."
        )
        torch.save(trainer.model.state_dict(), config.CHECKPOINT_PATH)
    else:
        print("Checkpoint found.")

    # 6. Verify Prediction/Submission
    print("\n--- Verifying Submission Generation ---")
    # Generate submission using the trained (or dummy) model
    generate_submission(config)

    if not os.path.exists(config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not generated.")

    # Load submission to check format
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission DataFrame Shape: {sub_df.shape}")
    print(f"Columns: {list(sub_df.columns)}")

    # Check columns
    if "id" not in sub_df.columns or "rle_mask" not in sub_df.columns:
        raise AssertionError("Submission missing required columns 'id' or 'rle_mask'.")

    # Check length (Test set has 1000 images)
    if len(sub_df) != 1000:
        raise AssertionError(f"Expected 1000 predictions, found {len(sub_df)}.")

    # Check content format (first row)
    first_rle = sub_df.iloc[0]["rle_mask"]
    # It can be NaN/empty string if mask is empty, or a string of numbers
    if pd.notna(first_rle) and first_rle != "":
        parts = first_rle.split()
        if len(parts) % 2 != 0:
            raise AssertionError("RLE string does not have pairs of values.")

    print("Submission format verified.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
