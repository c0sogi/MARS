import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import rle_encode, rle_decode, calc_iou
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import SaltNet
from library.losses import MultiTaskLoss
from library.engine import train_one_epoch, evaluate, generate_submission, set_seed


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration for fast demonstration...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Override Config defaults for speed
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SIZE = 32  # Use a tiny subset of data
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.CACHE_DIR = "./working/demo_cache"  # Isolate demo cache

    # Ensure working directories exist
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions (RLE)
    # -------------------------------------------------------------------------
    print("\n>>> Verifying RLE Encoding/Decoding logic...")

    # Create a synthetic mask (101x101) with a known pattern
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1  # A 10x10 square of salt

    # Encode
    encoded_str = rle_encode(dummy_mask)

    # Decode
    decoded_mask = rle_decode(encoded_str, shape=(101, 101))

    # Assert equality
    if not np.array_equal(dummy_mask, decoded_mask):
        raise AssertionError("RLE Decode does not match original mask!")

    print("RLE utilities verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Data Loading & Shape Verification
    # -------------------------------------------------------------------------
    print("\n>>> Initializing DataLoaders (Debug Mode)...")

    # Force reload to ensure debug settings apply
    train_loader, val_loader = get_dataloaders(fold=0, load_cached_data=False)

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))
    images, masks, depths, ids = batch

    print(
        f"Batch shapes -> Images: {images.shape}, Masks: {masks.shape}, Depths: {depths.shape}"
    )

    # Assertions for shapes
    # Images: (B, 1, 128, 128) - Note: 1 channel input, 128 padded size
    if images.shape != (Config.BATCH_SIZE, 1, Config.IMG_SIZE, Config.IMG_SIZE):
        raise AssertionError(f"Unexpected image shape: {images.shape}")

    # Masks: (B, 1, 128, 128)
    if masks.shape != (Config.BATCH_SIZE, 1, Config.IMG_SIZE, Config.IMG_SIZE):
        raise AssertionError(f"Unexpected mask shape: {masks.shape}")

    # Depths: (B, 1)
    if depths.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(f"Unexpected depth shape: {depths.shape}")

    print("DataLoader shapes verified.")

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass Verification
    # -------------------------------------------------------------------------
    print("\n>>> Initializing Model and Loss...")

    model = SaltNet().to(device)
    criterion = MultiTaskLoss(depth_weight=Config.DEPTH_LOSS_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Move batch to device
    images = images.to(device)
    masks = masks.to(device)
    depths = depths.to(device)

    # Forward pass
    print("Running dummy forward pass...")
    mask_logits, depth_pred = model(images)

    # Verify output shapes
    if mask_logits.shape != (Config.BATCH_SIZE, 1, Config.IMG_SIZE, Config.IMG_SIZE):
        raise AssertionError(f"Model output mask shape mismatch: {mask_logits.shape}")

    if depth_pred.shape != (Config.BATCH_SIZE, 1):
        raise AssertionError(f"Model output depth shape mismatch: {depth_pred.shape}")

    # CRITICAL: Verify depth head is connected to graph
    if not depth_pred.requires_grad:
        raise RuntimeError(
            "Depth prediction head is disconnected from computational graph!"
        )

    # Verify Loss calculation
    loss, metrics = criterion(mask_logits, depth_pred, masks, depths)

    if not loss.requires_grad:
        raise RuntimeError("Total loss does not require gradients!")

    print(f"Forward pass successful. Initial Loss: {loss.item():.4f}")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Executing Training Loop (1 Epoch)...")

    train_metrics = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch=1
    )

    print("\n>>> Executing Validation...")
    val_metrics = evaluate(model, val_loader, criterion, device)

    print(f"Training Demo Complete. Val mAP: {val_metrics['map']:.4f}")

    # -------------------------------------------------------------------------
    # 6. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n>>> Generating Submission for Test Set...")

    test_loader = get_test_dataloader(load_cached_data=False)
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    generate_submission(
        model, test_loader, device, output_path=submission_path, threshold=0.5
    )

    # Verify file creation
    if not os.path.exists(submission_path):
        raise FileNotFoundError("Submission file was not created.")

    # Verify content format
    df_sub = pd.read_csv(submission_path)
    if list(df_sub.columns) != ["id", "rle_mask"]:
        raise ValueError(f"Submission columns incorrect: {df_sub.columns}")

    print(f"Submission generated at {submission_path} with {len(df_sub)} rows.")
    print("\n>>> All tasks completed successfully.")


if __name__ == "__main__":
    run_demo()
