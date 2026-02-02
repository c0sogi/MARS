import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# Import library components
# We assume the library files are in the ./library directory relative to this script
from library.config import Config
from library.utils import seed_everything, get_logger, compute_metric
from library.dataset import get_dataloaders, BirdDataset
from library.model import MILResNet18
from library.engine import train_one_epoch, validate_one_epoch


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Setting up Demo Configuration...")

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 12  # Small subset for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Redirect working directories to a temporary demo folder
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Create these directories (Config usually does this on import, but we changed paths)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Setup reproducibility and logging
    seed_everything(Config.SEED)
    logger = get_logger(os.path.join(Config.WORKING_DIR, "demo.log"))
    logger("Demo configuration applied.")
    logger(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Test Utility Functions
    # ==========================================
    logger("\n[2] Testing Utility Functions...")

    # Test compute_metric (Macro ROC AUC)
    # Case: 2 samples, 3 classes
    y_true_dummy = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred_dummy = np.array([[0.9, 0.2, 0.8], [0.1, 0.8, 0.3]])
    auc_score = compute_metric(y_true_dummy, y_pred_dummy)

    logger(f"Computed Dummy AUC: {auc_score:.4f}")
    assert 0.0 <= auc_score <= 1.0, "AUC Score must be between 0 and 1"
    assert isinstance(auc_score, float), "AUC Score must be a float"

    # ==========================================
    # 3. Test Data Loading Pipeline
    # ==========================================
    logger("\n[3] Testing Data Loading Pipeline...")

    # Generate DataLoaders
    # We disable loading cached data to force the processing of raw BMPs
    train_loader, val_loader, test_loader = get_dataloaders(
        fold_idx=0, load_cached_data=False
    )

    logger(f"Train Loader Length: {len(train_loader)}")
    logger(f"Val Loader Length: {len(val_loader)}")

    # Fetch a single batch from training loader
    try:
        batch = next(iter(train_loader))
        inputs, labels, rec_ids = batch
    except StopIteration:
        raise RuntimeError(
            "DataLoader is empty! Check dataset paths and debug subset size."
        )

    # Validate Shapes
    # Expected Input: (Batch, Num_Tiles, Channels, Height, Width)
    # Config: Tiles=3, Channels=3, Size=224
    expected_input_shape = (
        Config.BATCH_SIZE,
        Config.NUM_TILES,
        Config.CHANNELS,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    expected_label_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)

    logger(f"Input Batch Shape: {inputs.shape}")
    logger(f"Label Batch Shape: {labels.shape}")

    assert (
        inputs.shape == expected_input_shape
    ), f"Input shape mismatch. Expected {expected_input_shape}, got {inputs.shape}"
    assert (
        labels.shape == expected_label_shape
    ), f"Label shape mismatch. Expected {expected_label_shape}, got {labels.shape}"

    # Validate Data Types and Ranges
    assert inputs.dtype == torch.float32, "Inputs should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"
    # Labels should be binary (0 or 1)
    assert torch.all((labels == 0) | (labels == 1)), "Labels must be binary"

    # ==========================================
    # 4. Test Model Architecture
    # ==========================================
    logger("\n[4] Testing Model Architecture...")

    model = MILResNet18()
    model.to(Config.DEVICE)

    # Perform a forward pass with the batch fetched earlier
    inputs = inputs.to(Config.DEVICE)
    with torch.no_grad():
        outputs = model(inputs)

    logger(f"Model Output Shape: {outputs.shape}")

    # Expected Output: (Batch, Num_Classes)
    assert outputs.shape == expected_label_shape, "Model output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"

    # ==========================================
    # 5. Test Training & Validation Engine
    # ==========================================
    logger("\n[5] Testing Training & Validation Engine...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Run one training epoch
    logger("Running training epoch...")
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, Config.DEVICE
    )
    logger(f"Train Loss: {train_loss:.4f}")

    assert not np.isnan(train_loss), "Training loss returned NaN"
    assert train_loss > 0, "Training loss should be positive"

    # Run one validation epoch
    logger("Running validation epoch...")
    val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, Config.DEVICE)
    logger(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    assert not np.isnan(val_loss), "Validation loss returned NaN"
    assert 0.0 <= val_auc <= 1.0, "Validation AUC out of range"

    logger("\n[6] Demo Execution Completed Successfully.")


if __name__ == "__main__":
    run_demo()
