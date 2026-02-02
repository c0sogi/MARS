import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import process_and_cache_data, IcebergDataset
from library.model import ShadowAwareWideBodyNet, train_one_epoch, validate


def run_demo():
    print("=== Starting Library Demonstration ===\n")

    # 1. Configuration Override for Speed
    # We modify the Config class attributes directly to ensure the demo runs quickly.
    print("1. Configuring environment for demo...")
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.NUM_FOLDS = 2  # (Not used in this single-loop demo, but good practice)
    Config.BATCH_SIZE = 16  # Small batch size
    Config.DEBUG = True  # Enable debug mode if applicable

    # Ensure reproducibility
    set_seed(Config.SEED)

    # Setup logger (prints to stdout)
    logger = setup_logger(os.path.join(Config.WORK_DIR, "demo_run.log"))
    logger.info(f"Device: {Config.DEVICE}")

    # 2. Data Loading and Verification
    print("\n2. Loading and Verifying Data...")
    # Load data using the library function
    data = process_and_cache_data(load_cached_data=True)

    # Verify Dictionary Keys
    expected_keys = [
        "X_train",
        "ang_train",
        "y_train",
        "ids_train",
        "X_val",
        "ang_val",
        "y_val",
        "ids_val",
        "X_test",
        "ang_test",
        "ids_test",
    ]
    for key in expected_keys:
        if key not in data:
            raise AssertionError(f"Missing key in data dictionary: {key}")

    # Verify Shapes
    # X_train should be (N, 3, 75, 75)
    X_train = data["X_train"]
    y_train = data["y_train"]
    ang_train = data["ang_train"]

    assert X_train.ndim == 4, f"Expected 4D image array, got {X_train.ndim}"
    assert X_train.shape[1] == 3, f"Expected 3 channels, got {X_train.shape[1]}"
    assert (
        X_train.shape[2] == 75 and X_train.shape[3] == 75
    ), "Expected 75x75 spatial dims"
    assert len(X_train) == len(y_train), "Mismatch between images and labels count"
    assert len(X_train) == len(ang_train), "Mismatch between images and angles count"

    print(f"   Data loaded successfully. Training samples: {len(X_train)}")
    print(f"   Image shape: {X_train.shape}")

    # 3. Dataset and DataLoader Instantiation
    print("\n3. Instantiating Dataset and DataLoader...")
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, data["ids_train"], transform=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple demo to avoid multiprocessing overhead
        pin_memory=True,
    )

    # Fetch one batch to verify
    images, angles, labels = next(iter(train_loader))

    # Verify Batch Shapes
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), f"Batch image shape mismatch: {images.shape}"
    assert angles.shape == (
        Config.BATCH_SIZE,
    ), f"Batch angle shape mismatch: {angles.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Batch label shape mismatch: {labels.shape}"

    print("   Batch fetched successfully.")
    print(
        f"   Batch Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # 4. Model Initialization
    print("\n4. Initializing Shadow-Aware Wide-Body Network...")
    model = ShadowAwareWideBodyNet().to(Config.DEVICE)

    # Verify model structure (basic check)
    # Check if SAAM modules exist
    assert hasattr(model, "saam1"), "Model missing SAAM block 1"
    assert hasattr(model, "pool1"), "Model missing DualPooling block 1"

    print("   Model instantiated and moved to device.")

    # 5. Forward Pass Verification
    print("\n5. Running Forward Pass Verification...")
    images = images.to(Config.DEVICE)
    angles = angles.to(Config.DEVICE)

    # Run forward pass
    outputs = model(images, angles)

    # Verify Output Shape: Should be (Batch_Size, 1)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch: {outputs.shape}"

    print(f"   Forward pass successful. Output shape: {outputs.shape}")

    # 6. Training Loop Demonstration
    print("\n6. Demonstrating Training Step (1 Epoch)...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for one epoch
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, Config.DEVICE
    )

    assert np.isfinite(train_loss), "Training loss is not finite (NaN or Inf)"
    print(f"   Training Epoch Completed. Loss: {train_loss:.6f}")

    # Validate
    print("   Running Validation...")
    # Create a small validation loader
    val_dataset = IcebergDataset(
        data["X_val"], data["ang_val"], data["y_val"], data["ids_val"], transform=False
    )
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    val_loss, val_probs, val_targets = validate(
        model, val_loader, criterion, Config.DEVICE
    )

    assert np.isfinite(val_loss), "Validation loss is not finite"
    assert len(val_probs) == len(data["X_val"]), "Validation predictions count mismatch"

    print(f"   Validation Completed. Loss: {val_loss:.6f}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
