import os
import torch
import pandas as pd
import numpy as np
import shutil
import sys

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.model import WIISNet
from library.dataset import get_dataloader
from library.trainer import Trainer
from library.inference import predict_and_submit


def run_demo():
    print("============================================================")
    print(" WIIS-Net Library Demonstration & Verification Script")
    print("============================================================")

    # ------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # ------------------------------------------------------------------
    print("\n[Step 1] Setting up demo configuration...")

    # Define a specific directory for this demo to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config class attributes for a fast, minimal run
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Point cache files to the demo directory
    Config.CACHE_TRAIN_IMAGES = os.path.join(demo_dir, "cache_train_images.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(demo_dir, "cache_train_targets.npy")
    Config.CACHE_TRAIN_IDS = os.path.join(demo_dir, "cache_train_ids.npy")

    Config.CACHE_VAL_IMAGES = os.path.join(demo_dir, "cache_val_images.npy")
    Config.CACHE_VAL_LABELS = os.path.join(demo_dir, "cache_val_targets.npy")
    Config.CACHE_VAL_IDS = os.path.join(demo_dir, "cache_val_ids.npy")

    Config.CACHE_TEST_IMAGES = os.path.join(demo_dir, "cache_test_images.npy")
    Config.CACHE_TEST_IDS = os.path.join(demo_dir, "cache_test_ids.npy")

    # Set Debug mode to process only a handful of subjects
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 6  # Process only 6 subjects

    # Training hyperparameters for speed
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Use 0 workers to avoid overhead in this short script
    Config.EARLY_STOPPING_PATIENCE = 1

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated for demo execution.")

    # ------------------------------------------------------------------
    # 2. Dataset & DataLoader Verification
    # ------------------------------------------------------------------
    print("\n[Step 2] Verifying Dataset and DataLoader...")

    # Initialize DataLoader (this triggers data processing and caching)
    # We set load_cached=False to force processing from metadata
    train_loader = get_dataloader("train", load_cached=False)

    # Fetch a single batch
    images, labels, subject_ids = next(iter(train_loader))

    print(
        f"Batch Shapes -> Images: {images.shape}, Labels: {labels.shape}, IDs: {subject_ids.shape}"
    )

    # Validation
    # Expected Image Shape: (Batch, 9, 224, 224)
    # 9 channels = 3 modalities * 3 slices per slab
    assert images.dim() == 4, "Images tensor must be 4D."
    assert (
        images.shape[1] == 9
    ), f"Expected 9 input channels (3 mods * 3 slices), got {images.shape[1]}."
    assert (
        images.shape[2] == Config.IMAGE_SIZE and images.shape[3] == Config.IMAGE_SIZE
    ), f"Image size mismatch. Expected {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}."

    # Expected Label Shape: (Batch, 1)
    assert (
        labels.dim() == 2 and labels.shape[1] == 1
    ), "Labels must be shape (Batch, 1)."

    print("Dataset logic verified successfully.")

    # ------------------------------------------------------------------
    # 3. Model Architecture Verification
    # ------------------------------------------------------------------
    print("\n[Step 3] Verifying WIISNet Architecture...")

    device = get_device()
    model = WIISNet().to(device)

    # Move sample batch to device
    images = images.to(device)

    # Forward pass
    outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")

    # Validation
    assert outputs.shape == (images.size(0), 1), "Model output shape mismatch."
    assert not torch.isnan(outputs).any(), "Model produced NaN values."

    print("Model architecture verified successfully.")

    # ------------------------------------------------------------------
    # 4. Training Loop Verification
    # ------------------------------------------------------------------
    print("\n[Step 4] Verifying Training Loop (Trainer)...")

    trainer = Trainer()

    # Run training for 1 epoch on the small debug dataset
    # We use load_cached_data=True because we generated the cache in Step 2
    trainer.fit(load_cached_data=True)

    # Validation: Check if the model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model checkpoint not found at {Config.MODEL_PATH}"
    print(f"Training loop completed. Model saved to {Config.MODEL_PATH}")

    # ------------------------------------------------------------------
    # 5. Inference Pipeline Verification
    # ------------------------------------------------------------------
    print("\n[Step 5] Verifying Inference Pipeline...")

    # Run the full prediction pipeline
    # This handles test data loading, prediction, consensus aggregation, and CSV saving
    predict_and_submit(load_cached_data=False)

    # Validation: Check submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    # Check columns
    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}."

    # Check if we have predictions (Config.DEBUG_SAMPLE_SIZE subjects)
    # Note: Test set size in debug mode might be limited by actual files available in test/
    assert len(df_sub) > 0, "Submission file is empty."

    print("Inference pipeline verified successfully.")

    print("\n============================================================")
    print(" DEMO COMPLETED SUCCESSFULLY")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
