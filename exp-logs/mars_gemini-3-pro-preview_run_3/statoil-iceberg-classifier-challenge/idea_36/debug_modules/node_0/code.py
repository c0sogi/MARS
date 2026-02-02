import os
import sys
import torch
import numpy as np
import pandas as pd

# 1. Setup Config for Speed and Isolation
# We import Config first to modify it before other modules use it.
from library.config import Config

# Override Config for Demo/Speed
# This ensures the implicit pipeline run in library.model executes quickly.
Config.EXP_ID = "demo_run"
Config.DEBUG = True
Config.MAX_DEBUG_SAMPLES = 64  # Small subset for speed
Config.EPOCHS = 1
Config.N_FOLDS = 2
Config.BATCH_SIZE = 8
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo

# Update paths to isolate this demo run
Config.WORKING_DIR = os.path.join(Config.BASE_DIR, "working", Config.EXP_ID)
Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
# Redirect submission to working dir
Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

# Create necessary directories
Config.setup()

print(
    f"Configuration updated: DEBUG={Config.DEBUG}, EPOCHS={Config.EPOCHS}, FOLDS={Config.N_FOLDS}"
)
print("Importing library.model (this will trigger the built-in training pipeline)...")
print("-" * 60)

# 2. Import Model (Triggers Pipeline due to code at end of library/model.py)
# We catch this execution as part of the demo, verifying the pipeline runs end-to-end.
import library.model
from library.model import DRHACNN

print("-" * 60)
print("Built-in pipeline completed. Starting explicit component demonstration.")
print("-" * 60)

# 3. Import other utilities
from library.dataset import get_loaders
from library.engine import train_one_epoch, validate, predict
from library.utils import set_seed, save_checkpoint, load_checkpoint


def demo_components():
    # Ensure reproducibility
    set_seed(42)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # A. Data Loading
    # -------------------------------------------------------------------------
    print("\n[A] Demonstrating Data Loading...")
    # get_loaders handles caching and dataset creation
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Verify Loaders
    assert len(train_loader) > 0, "Train loader is empty"
    assert len(val_loader) > 0, "Val loader is empty"

    # Fetch a batch to verify structure
    batch = next(iter(train_loader))
    images = batch["image"]
    angles = batch["angle"]
    labels = batch["label"]

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Image batch shape: {images.shape}")  # Should be (B, 3, 75, 75)
    print(f"Angle batch shape: {angles.shape}")  # Should be (B,)

    # Assertions
    assert images.dim() == 4, "Images should be 4D tensor (B, C, H, W)"
    assert images.size(1) == 3, "Images should have 3 channels"
    assert images.size(2) == 75 and images.size(3) == 75, "Images should be 75x75"
    assert angles.dim() == 1, "Angles should be 1D tensor"
    assert labels.dim() == 1, "Labels should be 1D tensor"

    # -------------------------------------------------------------------------
    # B. Model Instantiation
    # -------------------------------------------------------------------------
    print("\n[B] Demonstrating Model Instantiation...")
    model = DRHACNN().to(device)

    # Verify Forward Pass
    images = images.to(device)
    angles = angles.to(device)

    logits = model(images, angles)
    print(f"Logits shape: {logits.shape}")

    assert logits.dim() == 1, "Output logits should be 1D tensor"
    assert logits.size(0) == images.size(0), "Output batch size matches input"

    # -------------------------------------------------------------------------
    # C. Training Engine
    # -------------------------------------------------------------------------
    print("\n[C] Demonstrating Training Engine...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Train one epoch
    loss = train_one_epoch(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epoch=0,
        total_epochs=1,
    )
    print(f"Training Step Loss: {loss:.4f}")
    assert isinstance(loss, float), "Train loss should be a float"
    assert loss >= 0, "Loss should be non-negative"

    # Validate
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    assert isinstance(val_loss, float), "Val loss should be a float"

    # -------------------------------------------------------------------------
    # D. Inference
    # -------------------------------------------------------------------------
    print("\n[D] Demonstrating Inference...")
    ids, preds = predict(model, test_loader, device)

    print(f"Predictions shape: {preds.shape}")
    print(f"Sample predictions: {preds[:5]}")

    assert len(ids) == len(preds), "IDs and predictions count mismatch"
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions should be probabilities [0, 1]"

    # -------------------------------------------------------------------------
    # E. Checkpointing
    # -------------------------------------------------------------------------
    print("\n[E] Demonstrating Checkpointing...")

    # Save
    state = {
        "epoch": 1,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": val_loss,
    }
    save_checkpoint(state, is_best=True, checkpoint_dir=Config.CHECKPOINT_DIR, fold=0)

    expected_path = os.path.join(Config.CHECKPOINT_DIR, "checkpoint_fold_0.pth")
    assert os.path.exists(expected_path), "Checkpoint file was not created"

    # Load
    model_new = DRHACNN().to(device)
    checkpoint = load_checkpoint(expected_path, model_new, device=Config.DEVICE)

    print("Checkpoint loaded successfully.")
    assert "state_dict" in checkpoint, "Checkpoint missing state_dict"
    assert checkpoint["epoch"] == 1, "Checkpoint epoch mismatch"

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    demo_components()
