import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, rle_encode, dice_coef_metric
from library.dataset import ContrailDataset
from library.model import ConvNeXtHyperUNet
from library.loss import HybridLoss
from library.engine import train_one_epoch, valid_one_epoch


def run_demo():
    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print("[1/6] Setting up configuration and environment...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set deterministic behavior
    seed_everything(Config.SEED)

    # Override Config for a fast demonstration run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 12  # Small subset for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Setup directories
    Config.setup()

    device = Config.DEVICE
    print(f"      Device: {device}")
    print(f"      Debug Mode: {Config.DEBUG}")
    print(f"      Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # ==========================================
    # 2. Dataset & DataLoader Verification
    # ==========================================
    print("\n[2/6] Verifying Dataset and DataLoader...")

    # Initialize Datasets
    train_ds = ContrailDataset(split="train", debug=Config.DEBUG)
    valid_ds = ContrailDataset(split="validation", debug=Config.DEBUG)

    print(f"      Train Dataset Length: {len(train_ds)}")
    print(f"      Valid Dataset Length: {len(valid_ds)}")

    # Assert dataset is not empty
    if len(train_ds) == 0:
        raise ValueError("Train dataset is empty. Check metadata/input paths.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch one batch to verify shapes
    images, masks = next(iter(train_loader))

    print(f"      Input Batch Shape: {images.shape}")  # Expected: (B, 6, 256, 256)
    print(f"      Mask Batch Shape:  {masks.shape}")  # Expected: (B, 1, 256, 256)

    # Verify Shapes
    assert images.shape == (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Incorrect image shape: {images.shape}"
    assert masks.shape == (
        Config.BATCH_SIZE,
        Config.OUT_CHANNELS,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Incorrect mask shape: {masks.shape}"

    # Verify Data Range (Normalized roughly 0-1)
    print(f"      Image Max Value: {images.max():.4f}, Min Value: {images.min():.4f}")

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n[3/6] Initializing Model and checking Forward Pass...")

    model = ConvNeXtHyperUNet()
    model.to(device)

    # Run forward pass with the batch fetched earlier
    images = images.to(device, dtype=torch.float32)
    masks = masks.to(device, dtype=torch.float32)

    with torch.no_grad():
        logits = model(images)

    print(f"      Logits Shape: {logits.shape}")

    assert (
        logits.shape == masks.shape
    ), f"Output shape mismatch. Expected {masks.shape}, got {logits.shape}"

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n[4/6] Verifying Loss Function...")

    criterion = HybridLoss()

    # Calculate loss
    # Note: HybridLoss expects raw logits
    loss = criterion(logits, masks)

    print(f"      Calculated Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # ==========================================
    # 5. Training Loop Execution (Engine)
    # ==========================================
    print("\n[5/6] Executing Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scaler = torch.cuda.amp.GradScaler()

    # Train One Epoch
    print("      Training...")
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, device, scaler
    )
    print(f"      > Train Loss: {train_loss:.6f}")

    # Valid One Epoch
    print("      Validating...")
    val_loss, val_dice = valid_one_epoch(model, valid_loader, criterion, device)
    print(f"      > Valid Loss: {val_loss:.6f}")
    print(f"      > Global Dice: {val_dice:.6f}")

    # Basic sanity check on metrics
    assert train_loss > 0, "Train loss is zero or negative (unexpected for init)"

    # ==========================================
    # 6. Metric & Post-Processing Verification
    # ==========================================
    print("\n[6/6] Verifying Metrics and RLE Encoding...")

    # Create a synthetic mask: 10x10 square in a 256x256 grid
    dummy_pred = np.zeros((256, 256), dtype=np.float32)
    dummy_pred[50:60, 50:60] = 1.0  # High probability

    dummy_true = np.zeros((256, 256), dtype=np.float32)
    dummy_true[50:60, 50:60] = 1.0  # Perfect overlap

    # Test Dice Metric
    dice = dice_coef_metric(dummy_pred, dummy_true, threshold=0.5)
    print(f"      Perfect Overlap Dice: {dice:.4f}")
    assert np.isclose(dice, 1.0), f"Dice should be 1.0 for perfect overlap, got {dice}"

    # Test RLE Encoding
    # Flattened index logic:
    # (50,50) is pixel index 50 + 50*256 = 12850 (approx, depending on column-major order)
    # Task specifies column-major (top-to-bottom, then left-to-right)
    # Pixel (r, c) -> index = r + c * H + 1 (1-based)
    # Let's test a simple line: Pixel (0,0), (1,0), (2,0) -> Indices 1, 2, 3
    simple_mask = np.zeros((10, 10), dtype=np.uint8)
    simple_mask[0:3, 0] = 1  # Column 0, Rows 0-2

    encoded = rle_encode(simple_mask)
    print(f"      Simple RLE (3 pixels at start): '{encoded}'")

    # Expected: Start at 1, length 3 -> "1 3"
    assert encoded == "1 3", f"RLE Encoding failed. Expected '1 3', got '{encoded}'"

    print("\n==========================================")
    print("SUCCESS: All components verified.")
    print("==========================================")


if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"\nERROR: Script failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
