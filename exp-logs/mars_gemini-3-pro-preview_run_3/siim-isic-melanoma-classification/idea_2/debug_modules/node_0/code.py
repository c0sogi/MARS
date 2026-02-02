import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

# -----------------------------------------------------------------------------
# 1. Configuration & Setup
# -----------------------------------------------------------------------------
# Import config first to override settings for a fast, resource-efficient demo.
import library.config

# Modify configuration for demonstration purposes
library.config.DEBUG = True  # Enable debug mode to use a data subset
library.config.MAX_DEBUG_SAMPLES = 60  # Use only 60 samples for speed
library.config.EPOCHS = 1  # Run only 1 epoch
library.config.BATCH_SIZE = 8  # Small batch size
library.config.NUM_WORKERS = 0  # Avoid multiprocessing overhead
library.config.PRETRAINED = False  # Skip downloading weights
library.config.WORKING_DIR = "./working/demo_run"  # Isolated working dir

# Ensure working directory is clean to avoid loading stale cache
if os.path.exists(library.config.WORKING_DIR):
    shutil.rmtree(library.config.WORKING_DIR)
os.makedirs(library.config.WORKING_DIR, exist_ok=True)

# Import remaining library modules after config modification
# They will inherit the modified values from library.config
from library import utils, dataset, model, train

# -----------------------------------------------------------------------------
# 2. Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Starting Melanoma Classification Pipeline Demo ===")

    # A. Reproducibility
    utils.seed_everything(library.config.SEED)
    device = library.config.DEVICE
    print(f"Running on device: {device}")

    # B. Verify Utility Functions
    print("\n[1/6] Verifying Utils...")
    # Test ROC AUC
    y_true_test = [0, 0, 1, 1]
    y_pred_test = [0.1, 0.4, 0.6, 0.9]
    auc_score = utils.calculate_roc_auc(y_true_test, y_pred_test)
    print(f"  Calculated AUC: {auc_score}")
    assert 0.0 <= auc_score <= 1.0, "AUC must be between 0 and 1"

    # Test AverageMeter
    meter = utils.AverageMeter()
    meter.update(val=5.0, n=2)
    meter.update(val=10.0, n=2)  # Total sum 10+20=30, Total count 4
    assert meter.avg == 7.5, f"AverageMeter logic error. Expected 7.5, got {meter.avg}"
    print("  Utils verified.")

    # C. Verify Dataset and Dataloaders
    print("\n[2/6] Verifying Data Loading...")
    # load_cached_data=False forces the pipeline to process the raw CSVs (respecting DEBUG flag)
    train_loader, val_loader, test_loader, meta_dim = dataset.get_dataloaders(
        load_cached_data=False
    )

    print(f"  Meta Feature Dimension: {meta_dim}")
    assert meta_dim > 0, "Meta dimension should be positive."

    # Inspect one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    meta = batch["meta"]
    targets = batch["target"]

    print(f"  Batch Image Shape: {images.shape}")
    print(f"  Batch Meta Shape: {meta.shape}")

    assert images.shape == (
        library.config.BATCH_SIZE,
        3,
        library.config.IMG_SIZE,
        library.config.IMG_SIZE,
    )
    assert meta.shape == (library.config.BATCH_SIZE, meta_dim)
    assert targets.shape == (library.config.BATCH_SIZE,)
    print("  Data loading verified.")

    # D. Verify Model Architecture
    print("\n[3/6] Verifying Model...")
    # Initialize model (pretrained=False for speed)
    net = model.HybridEfficientNet(meta_dim=meta_dim, pretrained=False)
    net.to(device)

    # Forward pass check
    net.eval()
    with torch.no_grad():
        logits = net(images.to(device), meta.to(device))

    print(f"  Logits Shape: {logits.shape}")
    assert logits.shape == (
        library.config.BATCH_SIZE,
        1,
    ), "Model output shape mismatch."
    print("  Model verified.")

    # E. Verify Training and Validation Steps
    print("\n[4/6] Verifying Training & Validation Steps...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(net.parameters(), lr=1e-4)

    # Train one epoch
    train_loss, train_auc = train.train_one_epoch(
        train_loader, net, criterion, optimizer, device
    )
    print(f"  Train Loss: {train_loss:.4f} | Train AUC: {train_auc:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN."

    # Validate
    val_loss, val_auc = train.validate(val_loader, net, criterion, device)
    print(f"  Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN."
    print("  Training loop verified.")

    # F. Verify Checkpointing
    print("\n[5/6] Verifying Checkpointing...")
    ckpt_path = "demo_checkpoint.pth"

    # Save
    state = {
        "epoch": 1,
        "state_dict": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_score": val_auc,
    }
    utils.save_checkpoint(state, is_best=True, filename=ckpt_path)

    expected_path = os.path.join(library.config.WORKING_DIR, ckpt_path)
    assert os.path.exists(expected_path), "Checkpoint file was not created."

    # Load
    loaded_score = utils.load_checkpoint(net, optimizer, filename=ckpt_path)
    print(f"  Loaded Best Score: {loaded_score}")
    assert loaded_score == val_auc, "Loaded score does not match saved score."
    print("  Checkpointing verified.")

    # G. Verify Inference
    print("\n[6/6] Verifying Inference...")
    # Run inference using the loaded model
    submission_df = train.inference(test_loader, net, device)

    print(f"  Submission Shape: {submission_df.shape}")
    print(f"  Sample Predictions:\n{submission_df.head(3)}")

    # Checks
    assert "image_name" in submission_df.columns
    assert "target" in submission_df.columns
    assert len(submission_df) > 0
    assert os.path.exists(library.config.SUBMISSION_PATH), "Submission CSV not found."

    print("  Inference verified.")
    print("\n=== Demo Complete: All systems operational ===")
