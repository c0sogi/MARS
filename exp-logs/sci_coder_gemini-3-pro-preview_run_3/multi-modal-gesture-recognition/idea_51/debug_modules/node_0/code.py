import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.model import SKAGN
from library.trainer import Trainer, CascadedLoss


def main():
    # ==========================================
    # 1. Setup and Configuration Override
    # ==========================================
    print("Initializing demonstration...")
    set_seed(42)

    # Define a temporary working directory for this run
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Create "Mini" Metadata files to speed up execution
    # We take a small slice of the actual metadata
    print("Creating mini datasets for rapid testing...")

    # Load original metadata
    orig_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    orig_val = pd.read_csv(Config.VAL_METADATA_PATH)
    orig_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # Create mini versions (4 train, 2 val, 2 test samples)
    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")

    orig_train.head(4).to_csv(mini_train_path, index=False)
    orig_val.head(2).to_csv(mini_val_path, index=False)
    orig_test.head(2).to_csv(mini_test_path, index=False)

    # Override Config attributes globally to use the mini dataset and fast training settings
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    # Point caches to the demo directory
    Config.CACHE_TRAIN_PATH = os.path.join(demo_dir, "cache_train.npz")
    Config.CACHE_VAL_PATH = os.path.join(demo_dir, "cache_val.npz")
    Config.CACHE_TEST_PATH = os.path.join(demo_dir, "cache_test.npz")

    # Output paths
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Disable multiprocessing for tiny data
    Config.EARLY_STOPPING_PATIENCE = 2

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("Verifying Data Loader...")
    # This will process the mini metadata and create cache files
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch to verify shapes
    x_batch, y_batch = next(iter(train_loader))

    print(f"  Batch X shape: {x_batch.shape}")
    print(f"  Batch Y shape: {y_batch.shape}")

    # Assert correct dimensions: (Batch, Time, InputDim)
    assert x_batch.shape == (
        Config.BATCH_SIZE,
        Config.WINDOW_SIZE,
        Config.INPUT_DIM,
    ), f"Expected X shape {(Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.INPUT_DIM)}, got {x_batch.shape}"
    assert y_batch.shape == (
        Config.BATCH_SIZE,
        Config.WINDOW_SIZE,
    ), f"Expected Y shape {(Config.BATCH_SIZE, Config.WINDOW_SIZE)}, got {y_batch.shape}"

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = SKAGN().to(device)

    # Create dummy input
    dummy_input = torch.randn(
        Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.INPUT_DIM
    ).to(device)

    # Forward pass
    logits_1, logits_2, logits_3 = model(dummy_input)

    print(f"  Output shapes: {logits_1.shape}, {logits_2.shape}, {logits_3.shape}")

    expected_shape = (Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.NUM_CLASSES)
    assert logits_1.shape == expected_shape, "Stage 1 output shape mismatch"
    assert logits_2.shape == expected_shape, "Stage 2 output shape mismatch"
    assert logits_3.shape == expected_shape, "Stage 3 output shape mismatch"

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("Verifying Loss Calculation...")
    criterion = CascadedLoss().to(device)
    dummy_targets = torch.randint(
        0, Config.NUM_CLASSES, (Config.BATCH_SIZE, Config.WINDOW_SIZE)
    ).to(device)

    loss = criterion((logits_1, logits_2, logits_3), dummy_targets)
    print(f"  Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # ==========================================
    # 5. Training Loop Verification
    # ==========================================
    print("Verifying Training Loop...")
    # Initialize Trainer (will use the Config overrides and data loaders we verified)
    trainer = Trainer(load_cached_data=True)

    # Run fit
    trainer.fit(epochs=Config.EPOCHS)

    # Check if model checkpoint was saved
    assert os.path.exists(
        trainer.best_model_path
    ), "Best model checkpoint was not saved."
    print("  Training completed and model saved.")

    # ==========================================
    # 6. Submission Generation Verification
    # ==========================================
    print("Verifying Submission Generation...")
    trainer.generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Validate submission content format
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()

    print(f"  Generated {len(lines)} submission lines.")
    # Should match number of test samples in mini_test.csv (2)
    assert len(lines) == 2, f"Expected 2 submission lines, got {len(lines)}"

    # Check format of first line (SampleID,Labels...)
    sample_line = lines[0].strip().split(",")
    assert sample_line[0].startswith("Sample") or sample_line[0].startswith(
        "Session"
    ), "First column should be Sample ID"

    print("\nAll verification steps passed successfully!")


if __name__ == "__main__":
    main()
