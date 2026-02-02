import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the current directory is in the path to import library modules correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, rle_encode, dice_coefficient, load_metadata
from library.dataset import ContrailDataset, get_dataloaders, get_transforms
from library.model import DilatedResNetUNet
from library.loss import ContrailLoss
from library.train import train_model


def main():
    print("=== Contrail Detection Task: Library Usage Demonstration ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Setting up configuration...")

    # Define debug parameters for speed
    DEBUG_EPOCHS = 1
    DEBUG_BATCH_SIZE = 4
    DEBUG_MODE = True
    DEBUG_SAMPLE_SIZE = 12
    DEBUG_WORKERS = 2

    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Seed: {Config.SEED}")

    # --------------------------------------------------------------------------
    # 2. Dataset & DataLoader Verification
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Dataset and DataLoaders...")

    # Load metadata
    train_meta = load_metadata("train")
    assert not train_meta.empty, "Train metadata should not be empty."
    print(f"    Loaded {len(train_meta)} training records (metadata).")

    # Instantiate Dataset (subset)
    transform = get_transforms("train")
    dataset = ContrailDataset(train_meta.iloc[:5], split="train", transform=transform)

    # Fetch one sample
    image, mask = dataset[0]
    print(f"    Sample Image Shape: {image.shape} (Channels, H, W)")
    print(f"    Sample Mask Shape: {mask.shape} (Channels, H, W)")

    # Assertions
    # Image should be (3, 256, 256) for Ash composite
    assert image.shape == (3, 256, 256), f"Unexpected image shape: {image.shape}"
    assert mask.shape == (1, 256, 256), f"Unexpected mask shape: {mask.shape}"
    assert image.dtype == torch.float32, "Image dtype should be float32"
    assert mask.dtype == torch.float32, "Mask dtype should be float32"

    # Instantiate DataLoaders
    # We pass arguments explicitly to override Config defaults for this demo
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=DEBUG_BATCH_SIZE,
        num_workers=DEBUG_WORKERS,
        debug=DEBUG_MODE,
        debug_sample_size=DEBUG_SAMPLE_SIZE,
    )

    # Fetch one batch
    batch_imgs, batch_masks = next(iter(train_loader))
    print(f"    Batch Image Shape: {batch_imgs.shape}")
    print(f"    Batch Mask Shape: {batch_masks.shape}")

    assert batch_imgs.shape == (DEBUG_BATCH_SIZE, 3, 256, 256)
    assert batch_masks.shape == (DEBUG_BATCH_SIZE, 1, 256, 256)

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = DilatedResNetUNet().to(device)

    # Perform forward pass
    batch_imgs = batch_imgs.to(device)
    logits = model(batch_imgs)

    print(f"    Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        DEBUG_BATCH_SIZE,
        1,
        256,
        256,
    ), f"Expected logits shape ({DEBUG_BATCH_SIZE}, 1, 256, 256), got {logits.shape}"

    # --------------------------------------------------------------------------
    # 4. Loss Function Verification
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Loss Function...")

    criterion = ContrailLoss().to(device)
    batch_masks = batch_masks.to(device)

    loss = criterion(logits, batch_masks)
    print(f"    Calculated Loss: {loss.item():.6f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # --------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[5] Running Training Loop (Demo)...")

    # Run training with debug parameters
    best_dice = train_model(
        epochs=DEBUG_EPOCHS,
        batch_size=DEBUG_BATCH_SIZE,
        num_workers=DEBUG_WORKERS,
        debug=DEBUG_MODE,
        debug_sample_size=DEBUG_SAMPLE_SIZE,
        device=device,
    )

    print(f"    Training complete. Best Validation Dice: {best_dice:.6f}")

    # Verify checkpoint creation
    ckpt_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")
    assert os.path.exists(ckpt_path), f"Checkpoint not found at {ckpt_path}"
    print(f"    Checkpoint verified at: {ckpt_path}")

    # --------------------------------------------------------------------------
    # 6. Metric & Encoding Verification
    # --------------------------------------------------------------------------
    print("\n[6] Verifying Metrics and RLE Encoding...")

    # Test Dice Coefficient
    # Case: 50% overlap
    y_true = torch.tensor([1, 1, 0, 0]).float()
    y_pred = torch.tensor([1, 0, 1, 0]).float()
    # Intersection = 1 (index 0)
    # Union = sum(y_true) + sum(y_pred) = 2 + 2 = 4
    # Dice = 2*1 / 4 = 0.5
    dice_val = dice_coefficient(y_pred, y_true)
    print(f"    Test Dice (Expected 0.5): {dice_val:.4f}")
    assert np.isclose(
        dice_val, 0.5, atol=1e-5
    ), f"Dice calculation incorrect: {dice_val}"

    # Test Run-Length Encoding (RLE)
    # Create a 3x3 mask where the middle row is 1
    # Matrix:
    # 0 0 0
    # 1 1 1
    # 0 0 0
    # Flattened (Column-major/Fortran): 0, 1, 0, 0, 1, 0, 0, 1, 0
    # Indices (1-based):                1, 2, 3, 4, 5, 6, 7, 8, 9
    # Values:                           0, 1, 0, 0, 1, 0, 0, 1, 0
    # Runs of 1s:
    # Index 2 (len 1)
    # Index 5 (len 1)
    # Index 8 (len 1)
    # Expected RLE string: "2 1 5 1 8 1"

    mask_rle = np.zeros((3, 3), dtype=np.uint8)
    mask_rle[1, :] = 1
    encoded_str = rle_encode(mask_rle)
    print(f"    RLE Input (Middle Row 1s): \n{mask_rle}")
    print(f"    Encoded String: '{encoded_str}'")

    assert encoded_str == "2 1 5 1 8 1", f"RLE Encoding failed. Got: {encoded_str}"

    # Test Empty RLE
    empty_mask = np.zeros((5, 5), dtype=np.uint8)
    encoded_empty = rle_encode(empty_mask)
    assert (
        encoded_empty == "-"
    ), f"Empty mask should encode to '-', got '{encoded_empty}'"
    print("    Empty mask encoded correctly to '-'.")

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    main()
