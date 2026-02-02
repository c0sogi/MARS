import os
import sys
import shutil
import ast
import pandas as pd
import numpy as np
import torch
import torch.optim as optim

# Ensure library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, get_device, mcrmse_loss
from library.data import get_dataloaders, RNADataset
from library.model import DR_RHN
from library.train import train_epoch, validate


def create_mini_metadata(dest_dir):
    """
    Creates smaller versions of the metadata CSVs for rapid demonstration.
    """
    print(f"Creating mini metadata files in {dest_dir}...")

    # Paths to original metadata
    orig_train = "./metadata/train.csv"
    orig_val = "./metadata/val.csv"
    orig_test = "./metadata/test.csv"

    # Read top 20 rows
    # We read as object to preserve stringified lists
    df_train = pd.read_csv(orig_train).head(20)
    df_val = pd.read_csv(orig_val).head(20)
    df_test = pd.read_csv(orig_test).head(20)

    # Define new paths
    new_train_path = os.path.join(dest_dir, "mini_train.csv")
    new_val_path = os.path.join(dest_dir, "mini_val.csv")
    new_test_path = os.path.join(dest_dir, "mini_test.csv")

    # Save
    df_train.to_csv(new_train_path, index=False)
    df_val.to_csv(new_val_path, index=False)
    df_test.to_csv(new_test_path, index=False)

    return new_train_path, new_val_path, new_test_path


def configure_demo_settings(working_dir, train_csv, val_csv, test_csv):
    """
    Overrides Config attributes for the demo.
    """
    # Paths
    Config.WORKING_DIR = working_dir
    Config.TRAIN_CSV = train_csv
    Config.VAL_CSV = val_csv
    Config.TEST_CSV = test_csv

    # Caches - ensure unique names so we don't load old full data
    Config.TRAIN_CACHE = os.path.join(working_dir, "mini_train.npz")
    Config.VAL_CACHE = os.path.join(working_dir, "mini_val.npz")
    Config.TEST_CACHE = os.path.join(working_dir, "mini_test.npz")

    Config.BEST_MODEL_PATH = os.path.join(working_dir, "best_demo_model.pth")

    # Model Hyperparameters - Reduce for speed
    Config.BACKBONE_GROWTH_RATE = 16
    Config.BACKBONE_LAYERS = 2  # Minimal layers
    Config.LATENT_DIM = 16
    Config.FEEDBACK_HIDDEN_DIM = 8
    Config.FEEDBACK_EMBED_DIM = 8
    Config.RNN_HIDDEN_DIM = 16
    Config.RNN_LAYERS = 1

    # Training Hyperparameters
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script


def main():
    # 1. Setup
    seed_everything(42)
    device = get_device()
    print(f"Running on device: {device}")

    # Prepare working directory
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Create mini data
    train_csv, val_csv, test_csv = create_mini_metadata(demo_dir)

    # Apply Configuration
    configure_demo_settings(demo_dir, train_csv, val_csv, test_csv)

    # 2. Data Loading
    print("\n=== Loading Data ===")
    # load_cached_data=False ensures we process the new mini CSVs
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")

    # Verify Batch Content
    print("\n=== Verifying Batch Structure ===")
    batch = next(iter(train_loader))
    x, y, mask, p_idx, p_mask, ids = batch

    # Move to device for checking
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    p_idx, p_mask = p_idx.to(device), p_mask.to(device)

    print(f"Input (x) shape: {x.shape}")  # Expected: (B, 107, 18)
    print(f"Target (y) shape: {y.shape}")  # Expected: (B, 107, 5)
    print(f"Mask shape: {mask.shape}")  # Expected: (B, 107)
    print(f"Partner Index shape: {p_idx.shape}")

    # Assertions
    assert x.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_NODE_FEATURES,
    ), "Input shape mismatch"
    assert y.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), "Target shape mismatch"

    # 3. Model Instantiation
    print("\n=== Instantiating Model ===")
    model = DR_RHN().to(device)
    print("Model created.")

    # 4. Forward Pass & Loss Calculation
    print("\n=== Testing Forward Pass & Loss ===")
    # Forward
    y1, y2 = model(x, p_idx, p_mask)

    print(f"Output y1 shape: {y1.shape}")
    print(f"Output y2 shape: {y2.shape}")

    assert y2.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), "Output shape mismatch"

    # Loss
    loss = mcrmse_loss(y2, y, mask, Config.SCORED_TARGET_INDICES)
    print(f"Calculated MCRMSE Loss: {loss.item():.6f}")

    # 5. Training Loop Demonstration
    print("\n=== Running Training Loop Demo ===")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(Config.EPOCHS):
        # Note: library.train.train_epoch does NOT accept 'epoch' argument
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

    # 6. Inference Demonstration
    print("\n=== Running Inference Demo ===")
    model.eval()
    test_batch = next(iter(test_loader))
    tx, _, tmask, tp_idx, tp_mask, tids = test_batch

    tx = tx.to(device)
    tp_idx = tp_idx.to(device)
    tp_mask = tp_mask.to(device)

    with torch.no_grad():
        _, preds = model(tx, tp_idx, tp_mask)

    print(f"Test Batch Predictions Shape: {preds.shape}")
    print(f"Sample Prediction (First site, first target): {preds[0, 0, 0].item():.6f}")

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
