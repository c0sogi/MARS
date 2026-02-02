import sys
import os
import shutil
import torch
import numpy as np

# Ensure the current directory is in the path for library imports
sys.path.append(os.getcwd())

from library.utils import set_seed, get_device
from library.data_loader import get_loaders, get_test_loader
from library.model import MSD_SE_CNN
from library.train import train_kfold


def demo_pipeline():
    # --- Configuration ---
    # Use a specific directory for this demo to show caching behavior
    # and ensure we don't overwrite the main experiment files.
    DEMO_CACHE_DIR = "./working/demo_usage"
    BATCH_SIZE = 8
    DEBUG_MODE = True  # Uses only 100 samples for speed
    SEED = 42

    # Clean up previous demo run if exists to demonstrate data processing
    if os.path.exists(DEMO_CACHE_DIR):
        shutil.rmtree(DEMO_CACHE_DIR)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    print("=== Starting Library Usage Demo ===\n")

    # 1. Setup Device and Seed
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Loading (Debug Mode)
    print("\n--- 1. Data Loading & Preprocessing ---")
    print(f"Loading data with debug={DEBUG_MODE} (subset of 100 samples)...")

    # get_loaders handles loading raw data, processing bands, and creating K-Fold splits
    fold_loaders = get_loaders(
        batch_size=BATCH_SIZE,
        n_splits=5,
        seed=SEED,
        debug=DEBUG_MODE,
        cache_dir=DEMO_CACHE_DIR,
    )

    # Verify we got 5 folds
    if len(fold_loaders) != 5:
        raise AssertionError(f"Expected 5 folds, got {len(fold_loaders)}")

    # Extract the first fold's loaders
    train_loader, val_loader = fold_loaders[0]

    # Fetch one batch to inspect data structure
    images, angles, labels = next(iter(train_loader))

    print(f"Batch Shapes:")
    print(f"  Images: {images.shape} (Expected: [{BATCH_SIZE}, 3, 75, 75])")
    print(f"  Angles: {angles.shape} (Expected: [{BATCH_SIZE}, 1])")
    print(f"  Labels: {labels.shape} (Expected: [{BATCH_SIZE}, 1])")

    # Assertions to ensure logic correctness
    assert (
        images.dim() == 4 and images.shape[1] == 3
    ), "Image tensor must be (B, 3, H, W)"
    assert angles.dim() == 2, "Angles must be (B, 1)"
    assert labels.dim() == 2, "Labels must be (B, 1)"
    print("Data structure verified.")

    # 3. Model Instantiation & Forward Pass
    print("\n--- 2. Model Architecture ---")
    model = MSD_SE_CNN().to(device)

    # Move sample batch to device
    images_dev = images.to(device)
    angles_dev = angles.to(device)

    # Forward pass
    logits = model(images_dev, angles_dev)

    print(f"Model Output Shape: {logits.shape}")

    # Verify output
    assert logits.shape == (images.shape[0], 1), "Model output shape mismatch"
    assert logits.requires_grad, "Output tensor should require gradients"
    print("Model forward pass successful.")

    # 4. Training Loop Execution
    print("\n--- 3. Training Pipeline (Fast Check) ---")
    print("Running 5-Fold CV on debug subset (1 Epoch per fold)...")

    # train_kfold orchestrates the entire training process
    cv_scores = train_kfold(
        epochs=1,
        batch_size=BATCH_SIZE,
        patience=1,
        seed=SEED,
        debug=DEBUG_MODE,
        cache_dir=DEMO_CACHE_DIR,
    )

    print(f"CV Scores (Log Loss): {cv_scores}")
    assert len(cv_scores) == 5, "Training should return scores for 5 folds"
    assert all(isinstance(x, float) for x in cv_scores), "Scores must be floats"

    # 5. Inference Setup
    print("\n--- 4. Test Data Loader ---")
    test_loader = get_test_loader(batch_size=BATCH_SIZE, cache_dir=DEMO_CACHE_DIR)

    # Fetch test batch
    test_images, test_angles, test_ids = next(iter(test_loader))

    print(f"Test Batch ID Example: {test_ids[0]}")
    assert len(test_ids) == test_images.shape[0], "Mismatch between images and IDs"
    assert test_images.shape[1] == 3, "Test images should have 3 channels"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    demo_pipeline()
