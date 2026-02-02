import os
import torch
import pandas as pd
import numpy as np
import shutil

# 1. Import Config first to patch it for the demo
from library.config import Config

# ==========================================
# Configuration Patching for Demo Speed
# ==========================================
print("Patching Configuration for fast demonstration...")
Config.DEBUG = True
Config.DEBUG_SIZE = 6  # Very small subset for speed
Config.EPOCHS = 1  # Only 1 epoch
Config.BATCH_SIZE = 2  # Small batch size
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple demo

# Reduce dimensionality for speed
Config.NUM_SLICES = 8
Config.NUM_MODALITIES = 4
# Update dependent hyperparameters
Config.IN_CHANNELS = Config.NUM_SLICES * Config.NUM_MODALITIES
Config.STEM_GROUPS = (
    Config.NUM_SLICES
)  # Ensure groups match slices (4 channels per group)

# Set specific working directories for this demo
Config.CACHE_DIR = "./working/demo_run"
Config.SUBMISSION_FILE = "./working/demo_submission.csv"

# Ensure directories exist
os.makedirs(Config.CACHE_DIR, exist_ok=True)

# ==========================================
# Import Library Modules
# ==========================================
from library.config import set_seed
from library.utils import get_logger, get_device
from library.data_loader import get_dataloaders
from library.model import MGMT25DModel
from library.train import run_training, generate_submission

if __name__ == "__main__":
    # Setup
    logger = get_logger()
    logger.info("Starting MGMT Library Demonstration...")
    set_seed(Config.SEED)
    device = get_device()

    # ==========================================
    # 1. Data Loading Demonstration
    # ==========================================
    logger.info("\n[Step 1] Verifying Data Loading...")

    # Generate dataloaders
    # Note: This will process data and save to Config.CACHE_DIR
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Loader
    assert len(train_loader) > 0, "Train loader is empty"
    inputs, targets = next(iter(train_loader))

    # Expected shape: (B, IN_CHANNELS, H, W)
    expected_channels = Config.NUM_SLICES * Config.NUM_MODALITIES
    expected_shape = (
        Config.BATCH_SIZE,
        expected_channels,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )

    logger.info(f"Train Batch Input Shape: {inputs.shape}")
    logger.info(f"Train Batch Target Shape: {targets.shape}")

    # Assertions
    assert (
        inputs.shape == expected_shape
    ), f"Input shape mismatch. Expected {expected_shape}, got {inputs.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {targets.shape}"

    logger.info("Data Loading Verification Passed.")

    # ==========================================
    # 2. Model Architecture Demonstration
    # ==========================================
    logger.info("\n[Step 2] Verifying Model Architecture...")

    model = MGMT25DModel().to(device)
    inputs = inputs.to(device)

    # Forward pass
    logits = model(inputs)

    logger.info(f"Model Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert not torch.isnan(logits).any(), "Model produced NaN values"

    logger.info("Model Architecture Verification Passed.")

    # ==========================================
    # 3. Training Loop Demonstration
    # ==========================================
    logger.info("\n[Step 3] Running Training Loop (1 Epoch)...")

    # run_training handles the loop, saving best model, etc.
    best_model_path, returned_test_loader = run_training()

    logger.info(f"Training complete. Best model path: {best_model_path}")

    # Assertions
    assert os.path.exists(best_model_path), "Best model file was not saved."
    assert os.path.getsize(best_model_path) > 0, "Best model file is empty."

    logger.info("Training Loop Verification Passed.")

    # ==========================================
    # 4. Inference & Submission Demonstration
    # ==========================================
    logger.info("\n[Step 4] Generating Submission...")

    # Generate submission using the trained model
    generate_submission(model_path=best_model_path, test_loader=returned_test_loader)

    logger.info(f"Checking submission file at: {Config.SUBMISSION_FILE}")

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found."

    df = pd.read_csv(Config.SUBMISSION_FILE)
    logger.info(f"Submission Head:\n{df.head()}")

    # Assertions on content
    assert "BraTS21ID" in df.columns, "BraTS21ID column missing"
    assert "MGMT_value" in df.columns, "MGMT_value column missing"
    assert len(df) > 0, "Submission file is empty"
    assert (
        df["BraTS21ID"].dtype == int or df["BraTS21ID"].dtype == np.int64
    ), "BraTS21ID should be integer"
    assert (
        df["MGMT_value"].dtype == float or df["MGMT_value"].dtype == np.float64
    ), "MGMT_value should be float"

    # Check value range
    assert (
        df["MGMT_value"].min() >= 0.0 and df["MGMT_value"].max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    logger.info("Inference Verification Passed.")

    logger.info("\nAll demonstrations completed successfully.")
