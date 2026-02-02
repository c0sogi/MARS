import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Ensure library is in path
sys.path.append(os.getcwd())

# Import from provided library
from library.config import Config
from library.utils import set_seed
from library.model import CustomDenseNet
from library.data_loader import get_loaders, get_test_loader
from library.train import train_one_epoch, validate


def demonstration():
    print("=== Starting Demonstration of Iceberg Classification Library ===")

    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration...")

    # Modify Config for a fast demonstration run
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 50  # Use only 50 samples for speed
    Config.BATCH_SIZE = 10
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.NUM_FOLDS = 2  # Setup for 2 folds (we only run one)

    # Initialize environment
    Config.setup()
    set_seed(Config.SEED)

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    print("\n[2] Testing Data Loading...")

    # Get Train and Validation loaders for Fold 0
    # load_cached_data=True will use existing .npy files in working directory if available
    train_loader, val_loader = get_loaders(fold_idx=0, load_cached_data=True)

    # Verify Train Loader
    train_batch = next(iter(train_loader))
    images, angles, targets = train_batch

    print(f"    Train Batch Images Shape: {images.shape}")
    print(f"    Train Batch Angles Shape: {angles.shape}")
    print(f"    Train Batch Targets Shape: {targets.shape}")

    # Assertions for Train Loader
    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75), "Incorrect image dimensions"
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect angle dimensions"
    assert targets.shape == (Config.BATCH_SIZE,), "Incorrect target dimensions"
    assert images.dtype == torch.float32, "Images should be float32"

    # Verify Validation Loader
    val_batch = next(iter(val_loader))
    v_images, v_angles, v_targets = val_batch
    assert v_images.shape[1:] == (3, 75, 75), "Validation image dimensions mismatch"

    print("    Data Loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Testing Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = CustomDenseNet(
        growth_rate=12,
        block_config=(6, 12),  # Reduced config for demo speed
        num_init_features=24,
        drop_rate=0.1,
        fc_dim=64,
    ).to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward Pass
    logits = model(images, angles)

    print(f"    Model Output Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), "Model output shape should be (Batch, 1)"
    assert torch.isfinite(logits).all(), "Model output contains NaN or Inf"

    print("    Model Forward Pass verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Components
    # -------------------------------------------------------------------------
    print("\n[4] Testing Training Components...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train for one epoch
    print("    Running train_one_epoch...")
    train_loss = train_one_epoch(
        train_loader, model, criterion, optimizer, device, epoch=0
    )
    print(f"    Train Loss: {train_loss:.4f}")

    # Validate
    print("    Running validate...")
    val_loss = validate(val_loader, model, criterion, device)
    print(f"    Validation Loss: {val_loss:.4f}")

    # Assertions
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert isinstance(val_loss, float), "Validation loss should be a float"

    print("    Training components verification passed.")

    # -------------------------------------------------------------------------
    # 5. Inference Setup
    # -------------------------------------------------------------------------
    print("\n[5] Testing Inference Setup...")

    test_loader, test_ids = get_test_loader(load_cached_data=True)

    # Fetch one test batch
    t_images, t_angles = next(iter(test_loader))
    t_images = t_images.to(device)
    t_angles = t_angles.to(device)

    model.eval()
    with torch.no_grad():
        t_logits = model(t_images, t_angles)
        t_probs = torch.sigmoid(t_logits)

    print(f"    Test Batch Output Shape: {t_probs.shape}")

    # Assertions
    assert len(test_ids) > 0, "Test IDs should not be empty"
    assert t_probs.shape == (t_images.size(0), 1), "Test output shape mismatch"
    assert (t_probs >= 0).all() and (
        t_probs <= 1
    ).all(), "Probabilities must be between 0 and 1"

    print("    Inference setup verification passed.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    demonstration()
