import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_dataset_stats, get_device
from library.dataset import get_loaders
from library.model import get_model
from library.engine import fit, predict


def main():
    # --- 1. Setup and Configuration Overrides ---
    print("--- 1. Initializing Configuration ---")

    # Override Config defaults for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Use only 100 samples per split
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.PATIENCE = 1  # Minimal patience
    Config.NUM_WORKERS = 2  # Low worker count for simple demo

    # Initialize workspace directories
    Config.setup()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # Detect device
    device = get_device()
    print(f"Device: {device}")

    # --- 2. Dataset Statistics Calculation ---
    print("\n--- 2. Testing Dataset Statistics Calculation ---")

    # Force computation from scratch on a small subset
    mean, std = calculate_dataset_stats(
        metadata_path=Config.TRAIN_METADATA, sample_size=50, load_cached_data=False
    )

    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std:  {std}")

    # Verification
    assert isinstance(mean, np.ndarray), "Mean must be a numpy array"
    assert isinstance(std, np.ndarray), "Std must be a numpy array"
    assert mean.shape == (3,), f"Mean shape mismatch. Expected (3,), got {mean.shape}"
    assert std.shape == (3,), f"Std shape mismatch. Expected (3,), got {std.shape}"
    # Check if values are within reasonable normalized bounds [0, 1]
    assert np.all(mean >= 0.0) and np.all(mean <= 1.0), "Mean values out of bounds"

    # --- 3. Data Loading ---
    print("\n--- 3. Testing Data Loaders ---")

    train_loader, val_loader, test_loader = get_loaders(
        debug=Config.DEBUG,
        sample_size=Config.DEBUG_SAMPLE_SIZE,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    # Verification
    # Config.CROP_SIZE is 64. Expected shape: (B, 3, 64, 64)
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.CROP_SIZE, Config.CROP_SIZE)
    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"

    # Expected label shape: (B,)
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Label shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

    assert (
        labels.dtype == torch.float32
    ), "Labels should be float32 for BCEWithLogitsLoss"

    # --- 4. Model Initialization ---
    print("\n--- 4. Testing Model Initialization ---")

    model = get_model(device=device)

    # Perform a dummy forward pass
    with torch.no_grad():
        # Move inputs to device
        dummy_input = images.to(device)
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Verification
    # timm creates a model with num_classes=1, so output is (B, 1)
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {output.shape}"

    # --- 5. Training Loop Simulation ---
    print("\n--- 5. Testing Training Loop (Fit) ---")

    # Setup Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    # Execute Training
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
    )

    # Verify Checkpoint
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), f"Best model checkpoint not found at {Config.BEST_MODEL_PATH}"
    print("Checkpoint verification passed.")

    # --- 6. Inference Simulation ---
    print("\n--- 6. Testing Inference (Predict) ---")

    # Execute Prediction
    predict(model, test_loader, device)

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Rows: {len(df_sub)}")
    print(df_sub.head(3))

    # Verify Content
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "label" in df_sub.columns, "Submission missing 'label' column"
    # In debug mode, test set has DEBUG_SAMPLE_SIZE images
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(df_sub)}"

    print("\n--- All Tests Passed Successfully ---")


if __name__ == "__main__":
    main()
