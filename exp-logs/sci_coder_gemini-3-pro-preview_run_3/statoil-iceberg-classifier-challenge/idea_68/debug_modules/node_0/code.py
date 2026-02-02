import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data_loader import (
    load_data_and_cache,
    get_loaders,
    get_test_loader,
    IcebergDataset,
)
from library.model import RTICNN
from library.train import run_fold


def run_demo():
    print("=== Starting Demonstration of Iceberg Classifier Solution ===")

    # 1. Setup and Configuration Override
    print("\n[1] Configuring environment for fast demonstration...")

    # Set a separate working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)

    # Override Config parameters for speed and isolation
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Speed optimizations
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.NUM_EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Initialize directories
    Config.setup()
    Config.print_config()

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # 2. Data Pipeline Verification
    print("\n[2] Verifying Data Pipeline...")

    # Load data (this will process JSONs and save to the new cache dir)
    data = load_data_and_cache(load_cached_data=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    angle_train = data["angle_train"]

    # Assertions for data shape
    print(f"   Loaded X_train shape: {X_train.shape}")
    print(f"   Loaded y_train shape: {y_train.shape}")

    # Expected shape: (N, 3, 75, 75) where N is total samples
    assert len(X_train.shape) == 4, "X_train should be 4D"
    assert X_train.shape[1] == 3, "Should have 3 channels (HH, HV, Avg)"
    assert (
        X_train.shape[2] == 75 and X_train.shape[3] == 75
    ), "Image size should be 75x75"
    assert len(y_train) == len(X_train), "Label count mismatch"

    # Verify DataLoader
    print("   Testing DataLoader...")
    train_loader, val_loader = get_loaders(fold_idx=0, load_cached_data=True)

    # Fetch one batch
    images, angles, targets = next(iter(train_loader))

    print(f"   Batch images shape: {images.shape}")
    print(f"   Batch angles shape: {angles.shape}")
    print(f"   Batch targets shape: {targets.shape}")

    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75), "Incorrect batch image shape"
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect batch angle shape"
    assert targets.shape == (Config.BATCH_SIZE,), "Incorrect batch target shape"

    print("   Data Pipeline verified successfully.")

    # 3. Model Verification
    print("\n[3] Verifying Model Architecture (RTI-CNN)...")

    device = torch.device(Config.DEVICE)
    model = RTICNN().to(device)

    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model created with {param_count:,} trainable parameters.")

    # Forward pass check
    dummy_img = torch.randn(Config.BATCH_SIZE, 3, 75, 75).to(device)
    dummy_angle = torch.randn(Config.BATCH_SIZE).to(device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_img, dummy_angle)

    print(f"   Output shape: {output.shape}")

    # Output should be flattened logits (Batch_Size,)
    assert output.shape == (Config.BATCH_SIZE,), "Model output shape mismatch"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("   Model architecture verified successfully.")

    # 4. Training Loop Demonstration
    print("\n[4] Demonstrating Training Loop (Fold 0)...")

    # Run a short training session using the library function
    # This uses the modified Config (2 epochs, debug subset)
    best_val_loss = run_fold(fold_idx=0)

    print(f"   Training completed. Best Validation Loss: {best_val_loss:.4f}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_best_fold_0.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"   Checkpoint verified at: {checkpoint_path}")

    # 5. Inference Demonstration
    print("\n[5] Demonstrating Inference...")

    test_loader = get_test_loader(load_cached_data=True)

    # Load the best model from the training step
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print("   Running inference on first test batch...")
    with torch.no_grad():
        test_images, test_angles, test_ids = next(iter(test_loader))
        test_images = test_images.to(device)
        test_angles = test_angles.to(device)

        logits = model(test_images, test_angles)
        probs = torch.sigmoid(logits)

    print(f"   Predictions (first 5): {probs[:5].cpu().numpy()}")

    # Assertions
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"
    assert len(probs) == test_images.size(0), "Prediction count matches batch size"

    print("   Inference verified successfully.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
