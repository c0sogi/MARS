import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# Ensure the current directory is in the path so we can import the library
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import process_data, get_dataloaders, get_test_loader
from library.model import DIDPCNN
from library.train import Trainer


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("--- 1. Setting up Demo Configuration ---")

    # Override Config for a fast, lightweight demo run
    Config.IDEA_ID = "demo_usage"
    Config.WORKING_DIR = os.path.join("./working", Config.IDEA_ID)
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Speed optimizations
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.NUM_FOLDS = 2  # Only check logic for 2 folds (we will run 1 manually)
    Config.NUM_EPOCHS = 1  # Train for 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.DEVICE = (
        "cpu"  # Force CPU for simple logic verification if GPU is busy/overkill
    )
    if torch.cuda.is_available():
        Config.DEVICE = "cuda"

    # Setup directories
    Config.setup()

    # Set seeds
    seed_everything(Config.SEED)

    logger = get_logger("demo_script")
    logger.info(f"Running on device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Processing
    # ==========================================
    print("\n--- 2. Processing Data ---")
    # Force processing from scratch to verify logic (load_cached_data=False)
    # This will read from input/train.json and input/test.json, slice them (DEBUG mode), and save to cache
    data = process_data(load_cached_data=False)

    # Verify Data Shapes
    X_train = data["X_train"]
    y_train = data["y_train"]
    angle_train = data["angle_train"]

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"angle_train shape: {angle_train.shape}")

    # Assertions
    expected_size = Config.DEBUG_SUBSET_SIZE
    assert X_train.shape == (
        expected_size,
        3,
        75,
        75,
    ), f"Expected ({expected_size}, 3, 75, 75), got {X_train.shape}"
    assert y_train.shape == (
        expected_size,
    ), f"Expected ({expected_size},), got {y_train.shape}"
    assert angle_train.shape == (
        expected_size,
    ), f"Expected ({expected_size},), got {angle_train.shape}"
    assert not np.isnan(X_train).any(), "X_train contains NaNs"
    assert not np.isnan(
        angle_train
    ).any(), "angle_train contains NaNs (imputation failed)"

    # ==========================================
    # 3. DataLoader Verification
    # ==========================================
    print("\n--- 3. Verifying DataLoaders ---")
    train_loader, val_loader = get_dataloaders(
        data, fold_idx=0, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Fetch one batch
    images, angles, labels = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Angles Shape: {angles.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    # Assertions
    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75)
    assert angles.shape == (Config.BATCH_SIZE,)
    assert labels.shape == (Config.BATCH_SIZE,)
    assert images.dtype == torch.float32
    assert angles.dtype == torch.float32

    # ==========================================
    # 4. Model Initialization & Forward Pass
    # ==========================================
    print("\n--- 4. Model Forward Pass ---")
    model = DIDPCNN().to(Config.DEVICE)

    # Move batch to device
    images = images.to(Config.DEVICE)
    angles = angles.to(Config.DEVICE)

    # Forward
    outputs = model(images, angles)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert torch.isfinite(outputs).all(), "Model produced NaN or Inf values"

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n--- 5. Training Loop Execution ---")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(model, Config.DEVICE, criterion, optimizer)

    # Run one epoch
    train_loss = trainer.train_epoch(train_loader)
    print(f"Train Loss (1 epoch): {train_loss:.4f}")

    # Run validation
    val_loss = trainer.validate(val_loader)
    print(f"Val Loss: {val_loss:.4f}")

    assert isinstance(train_loss, float)
    assert isinstance(val_loss, float)
    assert train_loss > 0
    assert val_loss > 0

    # ==========================================
    # 6. Inference Demonstration
    # ==========================================
    print("\n--- 6. Inference Demonstration ---")
    test_loader = get_test_loader(
        data, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Get one test batch
    test_images, test_angles, test_ids = next(iter(test_loader))
    test_images = test_images.to(Config.DEVICE)
    test_angles = test_angles.to(Config.DEVICE)

    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(test_images, test_angles)
        probs = torch.sigmoid(logits)

    print(f"Test Batch IDs: {test_ids}")
    print(f"Predictions (Probs): {probs.cpu().numpy().flatten()}")

    assert len(probs) == Config.BATCH_SIZE
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
