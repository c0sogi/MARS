import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders, RSNADataset
from library.model import MultiStageConvNeXtMIL
from library.loss import RSNALoss
from library.engine import fit


def main():
    print("Initializing Demo Script...")

    # --- 1. Setup & Configuration Override ---
    # We modify Config parameters to ensure the demo runs quickly within the constraints.
    seed_everything(Config.SEED)

    # Create demo-specific directories in ./working
    DEMO_WORK_DIR = "./working/demo_run"
    DEMO_CACHE_DIR = "./working/demo_cache"
    os.makedirs(DEMO_WORK_DIR, exist_ok=True)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    # Override Config
    Config.CACHE_DIR = DEMO_CACHE_DIR
    Config.SUBMISSION_DIR = DEMO_WORK_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_WORK_DIR, "submission.csv")
    Config.MODEL_DIR = DEMO_WORK_DIR
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_WORK_DIR, "best_model.pth")

    # Reduce dimensions for speed
    Config.NUM_SLICES = 8  # Reduced from 64
    Config.IMAGE_SIZE = (128, 128)  # Reduced from 256
    Config.BATCH_SIZE = 2  # Small batch
    Config.EPOCHS = 1  # Single epoch
    Config.DEBUG = True  # Enable debug mode logic if applicable
    Config.DEBUG_SAMPLE_SIZE = 4  # Process only a few samples
    Config.NUM_WORKERS = 2  # Reduce worker overhead

    print("Configuration patched for demo execution.")

    # --- 2. Data Loading & Preparation ---
    print("Loading metadata...")
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Subset data manually to ensure we don't process too much
    # We select samples that definitely exist (checked by metadata generation)
    train_subset = train_df.iloc[:4].copy()
    val_subset = val_df.iloc[:2].copy()
    test_subset = test_df.iloc[:2].copy()

    # Save these subsets to working for reference (optional)
    train_subset.to_csv(os.path.join(DEMO_WORK_DIR, "demo_train.csv"), index=False)
    val_subset.to_csv(os.path.join(DEMO_WORK_DIR, "demo_val.csv"), index=False)

    print(
        f"Train subset: {len(train_subset)}, Val subset: {len(val_subset)}, Test subset: {len(test_subset)}"
    )

    # Initialize DataLoaders
    # Note: get_dataloaders handles transforms internally
    train_loader, val_loader, test_loader = get_dataloaders(
        train_subset, val_subset, test_subset
    )

    # --- 3. Verification: Dataset ---
    print("Verifying Dataset...")
    # Fetch one batch to verify shapes
    # RSNADataset returns (volume, targets)
    # Volume shape expected: (Batch, Channels, Height, Width)
    # where Channels = 3 (RGB/Stack) and the 'Slices' dim is handled by the model logic?
    # Wait, looking at library/data.py:
    # volume_reshaped = volume_tensor.view(N, C, H, W) where N=NUM_SLICES, C=3
    # Actually, the dataset returns: (N, C, H, W) per sample?
    # No, the DataLoader collates.
    # Let's check the dataset __getitem__ return:
    # It returns `volume_reshaped` which is (N, C, H, W).
    # So a batch from DataLoader will be (Batch, N, C, H, W).

    try:
        sample_batch, sample_targets = next(iter(train_loader))
        print(f"Batch Shape: {sample_batch.shape}")
        print(f"Target Shape: {sample_targets.shape}")

        expected_shape = (
            Config.BATCH_SIZE,
            Config.NUM_SLICES,
            3,
            Config.IMAGE_SIZE[0],
            Config.IMAGE_SIZE[1],
        )
        assert (
            sample_batch.shape == expected_shape
        ), f"Shape mismatch. Expected {expected_shape}, got {sample_batch.shape}"
        assert sample_targets.shape == (
            Config.BATCH_SIZE,
            Config.NUM_CLASSES,
        ), f"Target mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {sample_targets.shape}"
        print("Dataset verification passed.")
    except Exception as e:
        print(f"Dataset verification failed: {e}")
        raise e

    # --- 4. Verification: Model & Loss ---
    print("Verifying Model and Loss...")
    device = get_device()
    model = MultiStageConvNeXtMIL().to(device)
    criterion = RSNALoss()

    # Dummy forward pass
    # Input: (Batch, Slices, Channels, H, W)
    dummy_input = torch.randn(2, Config.NUM_SLICES, 3, 128, 128).to(device)
    dummy_targets = torch.zeros(2, 8).to(device)

    try:
        logits = model(dummy_input)
        loss = criterion(logits, dummy_targets)

        assert logits.shape == (
            2,
            8,
        ), f"Model output shape mismatch. Got {logits.shape}"
        assert not torch.isnan(loss), "Loss is NaN"
        print(f"Model forward pass successful. Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"Model/Loss verification failed: {e}")
        raise e

    # --- 5. Execution: Training Loop (Fit) ---
    print("Starting Training Loop (Fit)...")

    # We pass the subsets and loaders to the engine
    # The engine will run training, validation, and inference
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        test_df=test_subset,
        criterion=criterion,
        device=device,
    )

    # --- 6. Verification: Outputs ---
    print("Verifying Outputs...")

    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Check: Model file found at {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model file was not saved.")

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Check: Submission file found at {Config.SUBMISSION_PATH}")
        # Verify submission content
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        # Expected rows = 2 studies * 8 rows/study = 16 rows
        expected_rows = len(test_subset) * 8
        if len(sub_df) != expected_rows:
            raise AssertionError(
                f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"
            )
        print("Submission file content verified.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
