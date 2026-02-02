import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

# Import from the provided library
from library.config import Config
from library.utils import (
    seed_everything,
    rle_encode,
    rle_decode,
    compute_dice_coefficient,
    compute_hausdorff_3d,
)
from library.data import get_dataloaders
from library.model import build_model
from library.losses import CurriculumLoss
from library.engine import fit


def run_demo():
    # 1. Setup and Configuration
    print("--- Setting up Configuration ---")
    seed_everything(Config.SEED)

    # Override Config for a fast demonstration
    Config.DEBUG = True  # Use a small subset of data (300 train, 100 val)
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.VAL_BATCH_SIZE = 4
    Config.NUM_WORKERS = 2  # Limit workers for the demo environment
    Config.BACKBONE = "efficientnet_b0"  # Use smaller backbone if needed, but keeping b4 is fine for 1 epoch/small data

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    print("\n--- Initializing Data Loaders ---")
    train_loader, val_loader, test_loader = get_dataloaders(Config)

    # Verify Train Batch
    batch = next(iter(train_loader))
    images = batch["image"]
    masks = batch["mask"]
    ids = batch["id"]

    print(f"Batch Image Shape: {images.shape}")  # Expected: (B, 3, 320, 320)
    print(f"Batch Mask Shape: {masks.shape}")  # Expected: (B, 3, 320, 320)

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Image shape mismatch. Expected {(Config.BATCH_SIZE, Config.IN_CHANNELS, Config.IMG_SIZE[0], Config.IMG_SIZE[1])}, got {images.shape}"
    assert masks.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), "Mask shape mismatch."
    assert len(ids) == Config.BATCH_SIZE, "Batch ID count mismatch."

    print("Data loading verification passed.")

    # 3. Model Initialization
    print("\n--- Building Model ---")
    model = build_model(Config)
    model = model.to(Config.DEVICE)

    # Forward Pass Verification
    print("Running forward pass...")
    with torch.no_grad():
        # Move sample batch to device
        imgs_gpu = images.to(Config.DEVICE)
        outputs = model(imgs_gpu)

        # Check Deep Supervision output
        if Config.DEEP_SUPERVISION:
            assert isinstance(
                outputs, list
            ), "Model should return a list when deep_supervision is True."
            final_output = outputs[-1]
            print(f"Deep Supervision enabled. Number of outputs: {len(outputs)}")
        else:
            final_output = outputs

        assert final_output.shape == (
            Config.BATCH_SIZE,
            Config.NUM_CLASSES,
            Config.IMG_SIZE[0],
            Config.IMG_SIZE[1],
        ), f"Output shape mismatch. Got {final_output.shape}"

    print("Model forward pass verification passed.")

    # 4. Loss Function Verification
    print("\n--- Verifying Loss Function ---")
    loss_fn = CurriculumLoss(Config)

    # Compute loss on the sample batch
    # Note: outputs is a list if deep supervision is on
    masks_gpu = masks.to(Config.DEVICE)
    loss = loss_fn(outputs, masks_gpu, epoch=0)

    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() > 0, "Loss should be positive."

    print("Loss function verification passed.")

    # 5. Utility Verification
    print("\n--- Verifying Utilities ---")

    # Test RLE Encode/Decode
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1
    rle = rle_encode(dummy_mask)
    decoded = rle_decode(rle, (100, 100))
    assert np.array_equal(dummy_mask, decoded), "RLE Encode/Decode cycle failed."
    print("RLE Encode/Decode works correctly.")

    # Test Metrics
    # Perfect match
    dice_perfect = compute_dice_coefficient(dummy_mask, dummy_mask)
    assert (
        dice_perfect == 1.0
    ), f"Dice should be 1.0 for perfect match, got {dice_perfect}"

    # Empty match
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    dice_empty = compute_dice_coefficient(empty_mask, empty_mask)
    assert dice_empty == 0.0, f"Dice should be 0.0 for empty match, got {dice_empty}"

    # Hausdorff (3D)
    # Create simple 3D volumes (1, 100, 100)
    vol_a = dummy_mask[np.newaxis, :, :]
    vol_b = dummy_mask[np.newaxis, :, :]
    hd_perfect = compute_hausdorff_3d(vol_a, vol_b)
    assert (
        hd_perfect == 0.0
    ), f"Hausdorff should be 0.0 for perfect match, got {hd_perfect}"

    print("Metric functions verification passed.")

    # 6. Training Loop Execution
    print("\n--- Starting Training Loop (Demo) ---")

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Using a dummy scheduler for the demo (StepLR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Run the engine's fit function
    # This will train for 1 epoch on the debug subset and validate
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        config=Config,
    )

    print("\n--- Demo Completed Successfully ---")
    print(f"Check {Config.CHECKPOINT_DIR} for saved models.")


if __name__ == "__main__":
    run_demo()
