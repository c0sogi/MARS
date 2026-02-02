import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders, RSNADataset
from library.model import FractureMILModel
from library.loss import HierarchicalCompoundLoss
from library.engine import train_one_epoch, validate, inference


def main():
    print(
        "=== Starting Demonstration of Cervical Spine Fracture Detection Pipeline ==="
    )

    # 1. Configuration & Setup
    # Modify Config for speed and demonstration purposes
    print("\n[1] Configuring environment...")
    Config.IMAGE_SIZE = 64  # Reduce image size for speed
    Config.NUM_SLICES = 8  # Reduce depth for speed
    Config.BATCH_SIZE = 2  # Small batch size
    Config.EPOCHS = 1  # Single epoch
    Config.DEBUG_SAMPLE_SIZE = 6  # Use extremely small subset
    Config.CACHE_DIR = "./working/demo_cache"  # Separate cache for demo
    Config.SUBMISSION_PATH = "./working/submission.csv"

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Setup device
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Image Size: {Config.IMAGE_SIZE}")
    print(f"    Slices per Volume: {Config.NUM_SLICES}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            debug_sample_size=Config.DEBUG_SAMPLE_SIZE
        )
        print("    DataLoaders created successfully.")

        # Fetch one batch to verify shapes
        volumes, targets = next(iter(train_loader))

        # Expected Volume Shape: (Batch, Slices, Channels, Height, Width)
        # Channels = 3 (2.5D stacking)
        expected_vol_shape = (
            Config.BATCH_SIZE,
            Config.NUM_SLICES,
            3,
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        )
        assert (
            volumes.shape == expected_vol_shape
        ), f"Volume shape mismatch. Expected {expected_vol_shape}, got {volumes.shape}"

        # Expected Target Shape: (Batch, 8) -> 7 vertebrae + 1 patient_overall
        expected_target_shape = (Config.BATCH_SIZE, 8)
        assert (
            targets.shape == expected_target_shape
        ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

        print(f"    Batch Volume Shape: {volumes.shape}")
        print(f"    Batch Target Shape: {targets.shape}")
        print("    Data loading logic verified.")

    except Exception as e:
        print(f"    FATAL: Data loading failed: {e}")
        raise e

    # 3. Model Initialization & Forward Pass
    print("\n[3] Verifying Model Architecture...")
    try:
        model = FractureMILModel().to(device)
        print("    Model instantiated.")

        # Create dummy input on device
        dummy_input = torch.randn(expected_vol_shape).to(device)

        # Forward pass
        logits = model(dummy_input)

        # Expected Output: (Batch, 7) -> Logits for C1-C7
        expected_out_shape = (Config.BATCH_SIZE, 7)
        assert (
            logits.shape == expected_out_shape
        ), f"Model output shape mismatch. Expected {expected_out_shape}, got {logits.shape}"

        print(f"    Model Output Shape: {logits.shape}")
        print("    Model forward pass verified.")

    except Exception as e:
        print(f"    FATAL: Model verification failed: {e}")
        raise e

    # 4. Loss Function Verification
    print("\n[4] Verifying Hierarchical Compound Loss...")
    try:
        criterion = HierarchicalCompoundLoss()

        # Dummy logits (Batch, 7)
        dummy_logits = torch.randn((Config.BATCH_SIZE, 7)).to(device)

        # Dummy targets (Batch, 8) - float required for BCE
        dummy_targets = torch.randint(0, 2, (Config.BATCH_SIZE, 8)).float().to(device)

        loss = criterion(dummy_logits, dummy_targets)

        assert loss.dim() == 0, "Loss should be a scalar."
        assert not torch.isnan(loss), "Loss is NaN."
        assert loss.item() >= 0, "Loss should be non-negative."

        print(f"    Calculated Loss: {loss.item():.4f}")
        print("    Loss function verified.")

    except Exception as e:
        print(f"    FATAL: Loss verification failed: {e}")
        raise e

    # 5. Engine Execution (Train/Val/Inference)
    print("\n[5] Verifying Engine Execution...")

    # Optimizer setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Train one epoch
    print("    Running training step...")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"    Train Loss: {train_loss:.4f}")

    # Validate
    print("    Running validation step...")
    val_loss = validate(model, val_loader, criterion, device)
    print(f"    Val Loss: {val_loss:.4f}")

    # Inference
    print("    Running inference step...")
    inference(model, test_loader, device)

    # Check submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission file generated at {Config.SUBMISSION_PATH}")
        print(f"    Submission rows: {len(sub_df)}")
        print(f"    Submission columns: {list(sub_df.columns)}")

        # Basic check on submission content
        assert "row_id" in sub_df.columns and "fractured" in sub_df.columns
        assert len(sub_df) > 0
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
