import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd

# Import classes and functions from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import DeepHybridEfficientNet
from library.loss import FocalLoss
from library.train import train_one_epoch, valid_one_epoch, predict_test


def run_demo():
    print("=" * 40)
    print(" Running Melanoma Classification Demo")
    print("=" * 40)

    # ---------------------------------------------------------
    # 1. Configuration Setup for Speed/Debugging
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment...")

    # Override Config defaults for a quick demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use only 20 samples for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.WORK_DIR = "./working/demo_run"  # Separate working dir
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORK_DIR, "model_demo.pth")

    # Clean up demo directory if it exists
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 2. Data Loading Verification
    # ---------------------------------------------------------
    print("\n[Step 2] Loading DataLoaders...")

    # Force load_cached_data=False to verify metadata processing pipeline
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debug script to avoid overhead
        load_cached_data=False,
        debug=True,
    )

    # Check if loaders have data
    print(f"Train batches: {len(train_loader)}")
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert (
        len(train_loader.dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} samples, got {len(train_loader.dataset)}"

    # Fetch a single batch to inspect
    images, meta, targets = next(iter(train_loader))

    print(
        f"Batch Shapes -> Images: {images.shape}, Meta: {meta.shape}, Targets: {targets.shape}"
    )

    # Verify shapes
    # Images: (Batch, 3, H, W)
    assert images.shape == (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    # Meta: (Batch, Features)
    assert meta.ndim == 2
    # Targets: (Batch,)
    assert targets.shape == (Config.BATCH_SIZE,)

    meta_dim = meta.shape[1]
    print(f"Detected Metadata Dimension: {meta_dim}")

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[Step 3] Initializing DeepHybridEfficientNet...")

    # Initialize model (pretrained=False for faster init in demo)
    model = DeepHybridEfficientNet(meta_dim=meta_dim, pretrained=False)
    model = model.to(device)

    # Move data to device
    images = images.to(device)
    meta = meta.to(device)
    targets = targets.to(device)

    print("Performing dummy forward pass...")
    logits = model(images, meta)

    print(f"Output Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 1), "Output shape mismatch."

    # ---------------------------------------------------------
    # 4. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[Step 4] Verifying Focal Loss...")
    criterion = FocalLoss()
    loss = criterion(logits, targets)

    print(f"Calculated Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() >= 0, "Loss should be non-negative."

    # ---------------------------------------------------------
    # 5. Training Loop Simulation
    # ---------------------------------------------------------
    print("\n[Step 5] Running Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # No scheduler for this short demo

    train_loss, train_auc = train_one_epoch(
        model, train_loader, criterion, optimizer, scheduler=None, device=device
    )

    print(f"Train Result -> Loss: {train_loss:.4f} | AUC: {train_auc:.4f}")

    # Verify Validation Loop
    print("Running Validation Loop...")
    val_loss, val_auc = valid_one_epoch(model, val_loader, criterion, device)
    print(f"Valid Result -> Loss: {val_loss:.4f} | AUC: {val_auc:.4f}")

    # ---------------------------------------------------------
    # 6. Inference & Submission Generation
    # ---------------------------------------------------------
    print("\n[Step 6] Generating Predictions on Test Set...")

    image_names, preds = predict_test(model, test_loader, device)

    print(f"Generated {len(preds)} predictions.")
    print(f"First 3 predictions:\n{list(zip(image_names[:3], preds[:3]))}")

    # Verify predictions
    assert len(preds) == len(test_loader.dataset)
    assert len(image_names) == len(test_loader.dataset)
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions must be probabilities [0, 1]."

    print("\n" + "=" * 40)
    print(" Demo Completed Successfully")
    print("=" * 40)


if __name__ == "__main__":
    run_demo()
