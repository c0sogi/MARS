import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, rle_encode
from library.dataset import get_dataloader
from library.model import ContrailModel
from library.loss import ContrailLoss
from library.engine import train_one_epoch, validate, CheckpointManager

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("============================================================")
    print("       Contrail Identification: Pipeline Demonstration      ")
    print("============================================================")

    # --- 1. Configuration Setup ---
    print("\n[1] Setting up Configuration...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config parameters for a fast demo execution
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use a tiny subset of data
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.PREDICTION_DIR = os.path.join(Config.WORKING_DIR, "predictions")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")

    # Ensure directories exist (clean up if exists to start fresh)
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.PREDICTION_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # --- 2. Data Pipeline Verification ---
    print("\n[2] Verifying Data Pipeline...")

    try:
        # Initialize DataLoaders
        train_loader = get_dataloader(
            split="train", batch_size=Config.BATCH_SIZE, debug=True
        )
        val_loader = get_dataloader(
            split="validation", batch_size=Config.BATCH_SIZE, debug=True
        )

        print(f"    Train Loader Size: {len(train_loader)} batches")

        # Fetch a single batch
        images, masks = next(iter(train_loader))

        print(f"    Batch Images Shape: {images.shape}")
        print(f"    Batch Masks Shape: {masks.shape}")

        # Assertions
        expected_img_shape = (
            Config.BATCH_SIZE,
            Config.IN_CHANNELS,
            Config.IMG_SIZE,
            Config.IMG_SIZE,
        )
        expected_mask_shape = (Config.BATCH_SIZE, 1, Config.IMG_SIZE, Config.IMG_SIZE)

        assert (
            images.shape == expected_img_shape
        ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
        assert (
            masks.shape == expected_mask_shape
        ), f"Mask shape mismatch. Expected {expected_mask_shape}, got {masks.shape}"
        assert images.dtype == torch.float32, "Images should be float32"
        assert masks.dtype == torch.float32, "Masks should be float32"

        print("    >> Data Pipeline Verified Successfully.")

    except Exception as e:
        print(f"    !! Data Pipeline Failed: {e}")
        raise e

    # --- 3. Model Initialization & Forward Pass ---
    print("\n[3] Verifying Model Architecture...")

    try:
        model = ContrailModel().to(device)

        # Move batch to device
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        logits = model(images)
        print(f"    Logits Shape: {logits.shape}")

        assert logits.shape == expected_mask_shape, "Model output shape mismatch"
        print("    >> Model Architecture Verified Successfully.")

    except Exception as e:
        print(f"    !! Model Verification Failed: {e}")
        raise e

    # --- 4. Loss Function Verification ---
    print("\n[4] Verifying Loss Function...")

    try:
        criterion = ContrailLoss()
        loss = criterion(logits, masks)

        print(f"    Calculated Loss: {loss.item():.6f}")

        assert not torch.isnan(loss), "Loss is NaN"
        assert loss.item() >= 0, "Loss must be non-negative"
        print("    >> Loss Function Verified Successfully.")

    except Exception as e:
        print(f"    !! Loss Verification Failed: {e}")
        raise e

    # --- 5. Training Loop Simulation ---
    print("\n[5] Simulating Training Loop...")

    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        # AMP Scaler
        scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

        print("    Running Training Epoch 1...")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        print(f"    Epoch 1 Train Loss: {train_loss:.6f}")

        print("    Running Validation...")
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        print(f"    Validation Loss: {val_loss:.6f} | Dice: {val_dice:.6f}")

        assert train_loss >= 0, "Train loss should be valid"
        print("    >> Training Loop Verified Successfully.")

    except Exception as e:
        print(f"    !! Training Loop Failed: {e}")
        raise e

    # --- 6. Checkpoint Management ---
    print("\n[6] Verifying Checkpoint Manager...")

    try:
        ckpt_manager = CheckpointManager(Config.CHECKPOINT_DIR, top_k=2)

        # Save checkpoint
        saved_path = ckpt_manager.save(model, optimizer, epoch=1, score=val_dice)

        if saved_path:
            print(f"    Checkpoint saved at: {saved_path}")
            assert os.path.exists(saved_path), "Checkpoint file was not created"

        # Simulate a better epoch
        saved_path_2 = ckpt_manager.save(
            model, optimizer, epoch=2, score=val_dice + 0.1
        )
        print(f"    Better checkpoint saved at: {saved_path_2}")

        print("    >> Checkpoint Manager Verified Successfully.")

    except Exception as e:
        print(f"    !! Checkpoint Manager Failed: {e}")
        raise e

    # --- 7. Utility Verification (RLE) ---
    print("\n[7] Verifying Utilities (RLE Encoding)...")

    try:
        # Create a simple 4x4 mask with a vertical line in the first column
        # Indices (1-based, Col-Major): 1, 2, 3, 4 are the first column
        dummy_mask = np.zeros((4, 4), dtype=np.uint8)
        dummy_mask[:, 0] = 1

        encoded = rle_encode(dummy_mask)
        print(f"    Mask Shape: {dummy_mask.shape}")
        print(f"    Encoded String: '{encoded}'")

        # Expected: Start at 1, length 4. String: "1 4"
        assert (
            encoded == "1 4"
        ), f"RLE Encoding incorrect. Expected '1 4', got '{encoded}'"

        print("    >> RLE Utility Verified Successfully.")

    except Exception as e:
        print(f"    !! Utility Verification Failed: {e}")
        raise e

    print("\n============================================================")
    print("       Demo Execution Completed Successfully                ")
    print("============================================================")


if __name__ == "__main__":
    main()
