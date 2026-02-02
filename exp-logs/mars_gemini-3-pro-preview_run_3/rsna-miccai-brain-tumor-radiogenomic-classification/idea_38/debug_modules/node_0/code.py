import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import load_dataset
from library.data import get_dataloader
from library.model import S3DNet
from library.train import train_epoch, validate, set_seed


def run_demo():
    print(">>> Starting S3D-Net Pipeline Demo")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Define a temporary directory for this demo execution to avoid conflicts
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_execution")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"1. Configuration: Setting up temporary workspace at {demo_dir}")

    # Override Config parameters for speed and isolation
    Config.CACHE_DIR = demo_dir
    Config.CACHE_TRAIN_X = os.path.join(demo_dir, "X_train.npy")
    Config.CACHE_TRAIN_IDS = os.path.join(demo_dir, "ids_train.npy")
    Config.CACHE_TRAIN_Y = os.path.join(demo_dir, "y_train.npy")
    Config.CACHE_VAL_X = os.path.join(demo_dir, "X_val.npy")
    Config.CACHE_VAL_IDS = os.path.join(demo_dir, "ids_val.npy")
    Config.CACHE_VAL_Y = os.path.join(demo_dir, "y_val.npy")

    # Enable Debug mode to process only a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 4  # Process only 4 patients
    Config.BATCH_SIZE = 2
    Config.IMG_SIZE = 128  # Reduce image size for faster processing
    Config.NUM_EPOCHS = 1

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Loading & Processing
    # --------------------------------------------------------------------------
    print("\n2. Data Loading: Processing a subset of training data...")

    # Load training data (force processing from scratch by setting load_cached_data=False)
    # This tests library.utils.process_patient and library.utils.load_dataset
    X_train, y_train, ids_train = load_dataset("train", load_cached_data=False)

    # Validation assertions
    print(f"   Loaded Train X shape: {X_train.shape}")
    print(f"   Loaded Train y shape: {y_train.shape}")

    # Expected shape: (N, 2, 64, H, W) -> (4, 2, 64, 128, 128)
    assert X_train.ndim == 5, f"Expected 5D input tensor, got {X_train.ndim}"
    assert X_train.shape[0] == Config.DEBUG_SAMPLE_SIZE, "Sample size mismatch"
    assert X_train.shape[1] == 2, "Expected 2 streams (Even/Odd)"
    assert X_train.shape[2] == 64, "Expected 64 channels (16 slices * 4 modalities)"
    assert y_train.shape[0] == Config.DEBUG_SAMPLE_SIZE, "Target size mismatch"

    # Create a dummy validation set (reuse train data for demo purposes to save time)
    # We manually save it to the cache location so get_dataloader can pick it up as 'val'
    np.save(Config.CACHE_VAL_X, X_train)
    np.save(Config.CACHE_VAL_IDS, ids_train)
    np.save(Config.CACHE_VAL_Y, y_train)

    # --------------------------------------------------------------------------
    # 3. DataLoader Initialization
    # --------------------------------------------------------------------------
    print("\n3. DataLoader: Initializing and fetching a batch...")

    # Initialize DataLoader using the factory function
    train_loader = get_dataloader("train", load_cached_data=True)

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify batch structure
    assert "even" in batch
    assert "odd" in batch
    assert "target" in batch
    assert "BraTS21ID" in batch

    even_tensor = batch["even"]
    odd_tensor = batch["odd"]
    targets = batch["target"]

    print(f"   Batch 'even' tensor shape: {even_tensor.shape}")
    print(f"   Batch 'target' shape: {targets.shape}")

    assert even_tensor.shape == (
        Config.BATCH_SIZE,
        64,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    assert odd_tensor.shape == (Config.BATCH_SIZE, 64, Config.IMG_SIZE, Config.IMG_SIZE)

    # --------------------------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # --------------------------------------------------------------------------
    print("\n4. Model: Initializing S3DNet and running forward pass...")

    device = torch.device(Config.DEVICE)
    model = S3DNet()
    model.to(device)

    # Move batch to device
    even_input = even_tensor.to(device)
    odd_input = odd_tensor.to(device)

    # Forward pass
    logits = model(even_input, odd_input)

    print(f"   Output logits shape: {logits.shape}")

    assert logits.shape == (Config.BATCH_SIZE, 1), "Output shape mismatch"
    assert not torch.isnan(logits).any(), "Model produced NaN logits"

    # --------------------------------------------------------------------------
    # 5. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n5. Training: Running one epoch of training and validation...")

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Train for one epoch
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"   Train Loss: {train_loss:.4f}")

    assert isinstance(train_loss, float)
    assert train_loss >= 0, "Training loss should be non-negative"

    # Validate
    # We use the 'val' loader which we populated with the training data subset earlier
    val_loader = get_dataloader("val", load_cached_data=True)
    val_loss, val_auc = validate(model, val_loader, criterion, device)

    print(f"   Val Loss: {val_loss:.4f}")
    print(f"   Val AUC: {val_auc:.4f}")

    assert isinstance(val_auc, float)
    assert 0.0 <= val_auc <= 1.0, "AUC must be between 0 and 1"

    # --------------------------------------------------------------------------
    # 6. Cleanup
    # --------------------------------------------------------------------------
    print("\n6. Cleanup: Removing temporary files...")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
