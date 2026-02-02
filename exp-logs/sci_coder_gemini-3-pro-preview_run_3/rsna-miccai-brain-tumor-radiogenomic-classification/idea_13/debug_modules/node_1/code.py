import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import the provided library modules
from library import config
from library import utils
from library import data_loader
from library import model
from library import train
from library import predict


def run_demo():
    print("Initializing Demonstration...")

    # ==========================================
    # 1. Configuration Overrides for Speed
    # ==========================================
    # We modify the global config object to run a small-scale experiment
    # This ensures the demo completes quickly within the time limit.

    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Setting up working directory: {DEMO_DIR}")

    # Update paths
    config.CACHE_DIR = DEMO_DIR
    config.SUBMISSION_DIR = DEMO_DIR
    config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")
    config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    # Update hyperparameters for demo
    config.MAX_SAMPLES = 20  # Process only 20 samples per split
    config.BATCH_SIZE = 4  # Small batch size
    config.EPOCHS = 1  # Single epoch
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set seed for reproducibility
    utils.seed_everything(config.SEED)
    device = utils.get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loader Verification
    # ==========================================
    print("\n[Step 1] Verifying Data Loading...")

    # Force reload by ensuring no cache exists in the new demo dir initially
    # (Though the dir was just created, this is safe practice)
    if os.path.exists(os.path.join(DEMO_DIR, "cached_train_X.npy")):
        print("Cache found (unexpected for fresh demo), proceeding...")

    # Get dataloaders with the reduced sample size
    train_loader, val_loader, test_loader, test_ids = data_loader.get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=False,  # Force generation from metadata
        max_samples=config.MAX_SAMPLES,
    )

    # Verify Train Loader
    try:
        inputs, targets = next(iter(train_loader))
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    print(f"Batch Input Shape: {inputs.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Assertions for shapes
    # Expected: (B, 128, 256, 256)
    expected_channels = config.INPUT_CHANNELS  # 128
    expected_size = config.IMG_SIZE  # 256

    assert (
        inputs.shape[0] == config.BATCH_SIZE
    ), f"Expected batch size {config.BATCH_SIZE}, got {inputs.shape[0]}"
    assert (
        inputs.shape[1] == expected_channels
    ), f"Expected {expected_channels} channels, got {inputs.shape[1]}"
    assert (
        inputs.shape[2] == expected_size and inputs.shape[3] == expected_size
    ), f"Expected image size {expected_size}x{expected_size}, got {inputs.shape[2]}x{inputs.shape[3]}"
    assert targets.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Expected target shape ({config.BATCH_SIZE}, 1), got {targets.shape}"

    print("Data Loader verification passed.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[Step 2] Verifying Model Architecture...")

    net = model.MGMTNet().to(device)

    # Move the fetched batch to device
    inputs = inputs.to(device)

    # Forward pass
    with torch.no_grad():
        logits = net(inputs)

    print(f"Model Output Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected ({config.BATCH_SIZE}, 1), got {logits.shape}"

    # Check if output is not NaN
    if torch.isnan(logits).any():
        raise AssertionError("Model produced NaN values in forward pass.")

    print("Model architecture verification passed.")

    # ==========================================
    # 4. Training Execution
    # ==========================================
    print("\n[Step 3] Executing Training Loop (Demo)...")

    # We call the library's training function.
    # It uses the global config we modified earlier.
    train.run_training(max_samples=config.MAX_SAMPLES)

    # Verify model file creation
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Training failed to save model at {config.MODEL_SAVE_PATH}"
        )

    print("Training execution passed.")

    # ==========================================
    # 5. Inference Execution
    # ==========================================
    print("\n[Step 4] Executing Prediction Pipeline (Demo)...")

    # Run prediction using the trained model
    submission_df = predict.run_prediction(
        load_cached_data=True,  # Use the cache generated during data loading step if available
        max_samples=config.MAX_SAMPLES,
    )

    # Verify Submission DataFrame
    print("Verifying submission output...")
    print(submission_df.head())

    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"

    # Verify BraTS21ID format (should be strings or compatible)
    # The library returns IDs as they are in metadata (strings like '00013')
    # or integers depending on how they were loaded.
    # The sample submission usually expects IDs.

    # Verify values are probabilities (0-1)
    if not (
        (submission_df["MGMT_value"] >= 0) & (submission_df["MGMT_value"] <= 1)
    ).all():
        raise AssertionError("Predicted values are out of probability range [0, 1].")

    # Verify file existence
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    print("Prediction execution passed.")

    print("\n==========================================")
    print(" DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("==========================================")


if __name__ == "__main__":
    run_demo()
