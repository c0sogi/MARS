import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloader, WITSNetDataset
from library.model import WITSNet
from library.train import run_training
from library.inference import generate_submission


def main():
    print("Initializing WITS-Net Demo...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast demonstration
    Config.SEED = 42
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 6  # Use only 6 subjects for speed
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.LOAD_CACHED_DATA = False  # Force data processing to test pipeline logic

    # Use a specific directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    set_seed(Config.SEED)
    print(f"Configuration set. Working directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n--- Verifying Data Pipeline ---")

    # Load metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    if Config.DEBUG:
        df_train = df_train.head(Config.DEBUG_SUBSET_SIZE)

    # Create DataLoader
    print("Creating DataLoader...")
    train_loader = get_dataloader(
        df_train,
        mode="train",
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debugging to avoid multiprocessing overhead
    )

    # Fetch one batch
    images, targets, ids = next(iter(train_loader))

    # Assertions
    # Expected shape: (Batch, Channels, H, W) -> (2, 9, 224, 224)
    expected_channels = 9
    expected_size = 224

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        expected_channels,
        expected_size,
        expected_size,
    ), f"Incorrect image shape. Expected {(Config.BATCH_SIZE, expected_channels, expected_size, expected_size)}, got {images.shape}"

    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect target shape. Expected {(Config.BATCH_SIZE,)}, got {targets.shape}"

    print("Data pipeline verification successful.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n--- Verifying Model Architecture ---")

    model = WITSNet()

    # Check if the first layer was modified correctly for 9 channels
    first_layer = model.backbone.conv_stem
    print(f"First layer in_channels: {first_layer.in_channels}")

    assert (
        first_layer.in_channels == 9
    ), f"Model first layer should have 9 input channels, found {first_layer.in_channels}"

    # Test Forward Pass
    model.eval()
    with torch.no_grad():
        logits = model(images)

    print(f"Output Logits Shape: {logits.shape}")

    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    print("Model architecture verification successful.")

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\n--- Executing Training Loop (1 Epoch) ---")

    # run_training uses the Config settings we modified globally
    try:
        run_training()
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Check if model was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(
        best_model_path
    ), "Training finished but 'best_model.pth' was not found."
    print("Training loop completed and model saved.")

    # ==========================================
    # 5. Inference & Submission Generation
    # ==========================================
    print("\n--- Generating Submission ---")

    # Ensure test metadata exists (it should based on provided info)
    if not os.path.exists(Config.TEST_METADATA_PATH):
        raise FileNotFoundError(
            f"Test metadata not found at {Config.TEST_METADATA_PATH}"
        )

    # Run inference
    # Note: generate_submission relies on Config.TEST_METADATA_PATH and Config.SUBMISSION_PATH
    generate_submission(load_cached_data=False)

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")
    print(df_sub.head())

    # Check columns
    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check value range (probabilities)
    if not df_sub.empty:
        min_val = df_sub["MGMT_value"].min()
        max_val = df_sub["MGMT_value"].max()
        assert (
            0.0 <= min_val <= 1.0 and 0.0 <= max_val <= 1.0
        ), f"Predictions out of probability range [0, 1]. Min: {min_val}, Max: {max_val}"

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
