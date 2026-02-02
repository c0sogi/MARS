import os
import sys
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, compute_multilabel_auc
from library.data import get_dataloaders
from library.model import get_model
from library.engine import train_one_epoch, validate, SWAHandler, inference


def run_demo():
    print("Initializing Demo...")

    # 1. Configure for Speed/Debug
    # We modify the Config singleton directly to run a fast verification
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Small subset for quick testing
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")

    # Ensure clean slate for demo cache
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    device = Config.get_device()
    print(f"Device: {device}")

    # =========================================================================
    # DEMO: Data Loading
    # =========================================================================
    print("\n[1/5] Testing Data Loading...")

    # Load dataloaders with debug=True to use the small subset
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False  # Force re-creation of cache
    )

    # Verification
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"

    # Fetch one batch to verify shapes
    images, targets = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Expected: (Batch_Size, 3, Height, Width)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    ), f"Incorrect image shape: {images.shape}"
    # Expected: (Batch_Size, Num_Classes)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Incorrect target shape: {targets.shape}"

    # Verify cache creation
    cache_files = os.listdir(Config.CACHE_DIR)
    assert any(
        "metadata_cache_debug" in f for f in cache_files
    ), "Metadata cache file was not created."
    print("Data loading and caching verified.")

    # =========================================================================
    # DEMO: Model Instantiation
    # =========================================================================
    print("\n[2/5] Testing Model Instantiation...")

    # Use a lightweight backbone for the demo (resnet18)
    model = get_model(
        model_name="resnet18", num_classes=Config.NUM_CLASSES, pretrained=False
    )
    model = model.to(device)

    # Test forward pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1]).to(
            device
        )
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {output.shape}"
    print("Model instantiation verified.")

    # =========================================================================
    # DEMO: Training Loop (One Epoch)
    # =========================================================================
    print("\n[3/5] Testing Training Loop...")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train for one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=1)

    print(f"Train Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"
    print("Training loop verified.")

    # =========================================================================
    # DEMO: Validation & Metrics
    # =========================================================================
    print("\n[4/5] Testing Validation...")

    val_loss, val_auc = validate(model, val_loader, device)

    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val AUC: {val_auc:.4f}")

    assert isinstance(val_loss, float), "Val loss should be a float"
    assert 0.0 <= val_auc <= 1.0, "AUC score out of range [0, 1]"
    print("Validation verified.")

    # =========================================================================
    # DEMO: SWA Handler & Inference
    # =========================================================================
    print("\n[5/5] Testing SWA Handler and Inference...")

    # Initialize SWA Handler with start_epoch=0 so it activates immediately for demo
    swa_handler = SWAHandler(model, swa_start_epoch=0, device=device)

    # Simulate end of an epoch
    swa_handler.on_epoch_end(model, epoch=0)

    assert swa_handler.active, "SWA Handler should be active after start epoch"

    # Get SWA model
    swa_model = swa_handler.get_model()
    assert swa_model is not None, "SWA model should be available"

    # Finalize SWA (update BN stats)
    # We use train_loader for BN update as per standard practice
    swa_handler.finalize(train_loader)

    # Test Inference with SWA model
    # Using Test Time Augmentation (TTA) = True
    preds = inference(swa_model, test_loader, device, use_tta=True)

    print(f"Inference Predictions Shape: {preds.shape}")

    # Expected: (Num_Test_Samples, Num_Classes)
    # Note: len(test_loader.dataset) depends on DEBUG_SUBSET_SIZE
    expected_rows = len(test_loader.dataset)
    assert preds.shape == (
        expected_rows,
        Config.NUM_CLASSES,
    ), f"Inference shape mismatch. Expected ({expected_rows}, {Config.NUM_CLASSES}), got {preds.shape}"

    # Check probabilities range
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions should be probabilities in [0, 1]"

    print("SWA and Inference verified.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
