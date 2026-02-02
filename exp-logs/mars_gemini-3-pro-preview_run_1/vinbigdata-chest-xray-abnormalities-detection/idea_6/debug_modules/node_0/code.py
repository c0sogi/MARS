import sys
import os
import types
import torch
import numpy as np
import pandas as pd
import warnings
from torch.utils.data import DataLoader

# =========================================================================
# 0. Environment Setup & TQDM Suppression
# =========================================================================
# Suppress warnings
warnings.filterwarnings("ignore")


# Mock tqdm to suppress progress bar output as per requirements
class MockTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


# Replace the tqdm module in sys.modules so 'from tqdm import tqdm' works silently
mock_tqdm_module = types.ModuleType("tqdm")
mock_tqdm_module.tqdm = MockTqdm
sys.modules["tqdm"] = mock_tqdm_module

# Now import library modules
from library.config import Config
from library.utils import get_image_and_dimensions
from library.dataset import ThoracicDataset
from library.model import ThoracicModel
from library.loss import ThoracicLoss
from library.engine import train_one_epoch, validate, generate_submission


def run_demo():
    print("Starting Thoracic Disease Detection Demo...")

    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("\n[1] Setting up Configuration...")

    # Configure for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 16  # Small subset for quick execution
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Setup directories
    Config.setup()

    # Set reproducibility seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print(f"  Device: {Config.DEVICE}")
    print(f"  Debug Mode: {Config.DEBUG}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")

    # =========================================================================
    # 2. Dataset & DataLoader Verification
    # =========================================================================
    print("\n[2] Verifying Dataset and DataLoader...")

    # Initialize Datasets
    train_dataset = ThoracicDataset(mode="train", debug=True)
    val_dataset = ThoracicDataset(mode="val", debug=True)

    print(f"  Train Dataset Size: {len(train_dataset)}")
    print(f"  Val Dataset Size: {len(val_dataset)}")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch one batch to verify structure
    images, targets, image_ids = next(iter(train_loader))

    # Assertions for Shapes
    print("  Verifying batch shapes...")

    # Image: (B, 3, H, W)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect image shape: {images.shape}"

    # Calculate expected output size based on downsample ratio
    expected_out_size = Config.IMG_SIZE // Config.DOWNSAMPLE_RATIO
    num_finding_classes = Config.NUM_CLASSES - 1

    # Heatmap: (B, NumFindings, H_out, W_out)
    assert targets["heatmap"].shape == (
        Config.BATCH_SIZE,
        num_finding_classes,
        expected_out_size,
        expected_out_size,
    ), f"Incorrect heatmap shape: {targets['heatmap'].shape}"

    # Size & Offset: (B, 2, H_out, W_out)
    assert targets["size"].shape == (
        Config.BATCH_SIZE,
        2,
        expected_out_size,
        expected_out_size,
    ), f"Incorrect size target shape: {targets['size'].shape}"
    assert targets["offset"].shape == (
        Config.BATCH_SIZE,
        2,
        expected_out_size,
        expected_out_size,
    ), f"Incorrect offset target shape: {targets['offset'].shape}"

    # Global Label: (B, 1)
    assert targets["global_label"].shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect global label shape: {targets['global_label'].shape}"

    print("  Dataset verification passed.")

    # =========================================================================
    # 3. Model Initialization & Forward Pass
    # =========================================================================
    print("\n[3] Initializing Model and checking Forward Pass...")

    device = Config.DEVICE
    model = ThoracicModel().to(device)

    # Move batch to device
    images = images.to(device)

    # Forward pass
    outputs = model(images)

    # Verify Output Shapes
    assert outputs["heatmap"].shape == (
        Config.BATCH_SIZE,
        num_finding_classes,
        expected_out_size,
        expected_out_size,
    ), "Model heatmap output shape mismatch"

    assert outputs["size"].shape == (
        Config.BATCH_SIZE,
        2,
        expected_out_size,
        expected_out_size,
    ), "Model size output shape mismatch"

    assert outputs["global_prob"].shape == (
        Config.BATCH_SIZE,
        1,
    ), "Model global_prob output shape mismatch"

    print("  Model forward pass successful.")

    # =========================================================================
    # 4. Loss Calculation
    # =========================================================================
    print("\n[4] Verifying Loss Calculation...")

    criterion = ThoracicLoss()

    # Move targets to device
    targets_device = {k: v.to(device) for k, v in targets.items()}

    loss, loss_stats = criterion(outputs, targets_device)

    print(f"  Calculated Loss: {loss.item():.4f}")

    # Basic sanity checks
    assert not torch.isnan(loss), "Loss is NaN!"
    assert loss.item() > 0, "Loss should be positive"

    print("  Loss verification passed.")

    # =========================================================================
    # 5. Training Loop (Engine)
    # =========================================================================
    print("\n[5] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    # Train for one epoch
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, device, epoch=0
    )

    # Validate
    val_loss = validate(model, val_loader, device)

    print(f"  Epoch 1 Train Loss: {train_loss:.4f}")
    print(f"  Epoch 1 Val Loss: {val_loss:.4f}")

    # =========================================================================
    # 6. Inference & Submission
    # =========================================================================
    print("\n[6] Generating Submission...")

    # Setup Test Dataset
    test_dataset = ThoracicDataset(mode="test", debug=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Generate predictions
    df_sub = generate_submission(model, test_loader, device)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created!"

    # Verify Content
    assert len(df_sub) == len(
        test_dataset
    ), f"Submission rows ({len(df_sub)}) do not match test set size ({len(test_dataset)})"

    expected_cols = ["image_id", "PredictionString"]
    assert all(
        col in df_sub.columns for col in expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    print("  Submission generated successfully.")
    print("  Head of submission:")
    print(df_sub.head())

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
