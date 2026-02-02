import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.model import CMSDI_CNN
from library.data import get_data_loaders
from library.train import train_epoch, validate


def run_demo():
    print("=== Starting CMSDI-CNN Pipeline Demonstration ===\n")

    # 1. Setup and Configuration Overrides for Speed
    print("[1] Configuring environment...")
    seed_everything(42)
    device = get_device()
    print(f"    Device: {device}")

    # Override Config for a quick demo run
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = (
        0  # Use main process for data loading to avoid overhead in demo
    )
    Config.NUM_DROPOUT_SAMPLES = 5  # Keep default, but explicit for checking

    # Ensure working directories exist (Config does this on import, but good practice)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading...")
    # We use fold 0 for demonstration. load_cached_data=False ensures we test the processing logic once.
    # In a real run, we would likely use True.
    train_loader, val_loader = get_data_loaders(fold=0, load_cached_data=False)

    # Fetch one batch to inspect
    images, angles, labels = next(iter(train_loader))

    print(
        f"    Batch shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions for Data
    # Images: [Batch, 3, 75, 75]
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Incorrect image tensor shape"
    # Angles: [Batch]
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect angle tensor shape"
    # Labels: [Batch]
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label tensor shape"
    assert images.dtype == torch.float32, "Images should be float32"

    print("    Data Loading verification passed.")

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture (CMSDI_CNN)...")
    model = CMSDI_CNN().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Test Forward Pass in TRAINING mode (Multi-Sample Dropout)
    model.train()
    logits_train = model(images, angles)
    print(f"    Training Output Shape: {logits_train.shape}")

    # Expectation: [Batch, Num_Samples]
    assert logits_train.shape == (
        Config.BATCH_SIZE,
        Config.NUM_DROPOUT_SAMPLES,
    ), f"Expected train output shape {(Config.BATCH_SIZE, Config.NUM_DROPOUT_SAMPLES)}, got {logits_train.shape}"

    # Test Forward Pass in EVAL mode (Averaged Logits)
    model.eval()
    with torch.no_grad():
        logits_eval = model(images, angles)
    print(f"    Inference Output Shape: {logits_eval.shape}")

    # Expectation: [Batch, 1]
    assert logits_eval.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected eval output shape {(Config.BATCH_SIZE, 1)}, got {logits_eval.shape}"

    print("    Model architecture verification passed.")

    # 4. Training Loop Demonstration
    print("\n[4] Running Training Loop (1 Epoch)...")

    # Re-initialize model and optimizer
    model = CMSDI_CNN().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run one training epoch
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"    Train Loss: {train_loss:.6f}")

    # Validate
    val_loss = validate(model, val_loader, device)
    print(f"    Validation Loss: {val_loss:.6f}")

    # Assertions for Loss
    assert isinstance(train_loss, float), "Train loss should be a float"
    assert isinstance(val_loss, float), "Validation loss should be a float"
    assert not np.isnan(train_loss), "Train loss is NaN"
    assert not np.isnan(val_loss), "Validation loss is NaN"

    print("    Training loop verification passed.")

    # 5. Inference / Prediction Check
    print("\n[5] Simulating Prediction...")
    model.eval()
    with torch.no_grad():
        # Use the validation batch from earlier
        # Move labels to device for consistency, though not needed for prediction
        labels = labels.to(device)

        raw_logits = model(images, angles)
        probs = torch.sigmoid(raw_logits)

        print(f"    Raw Logits (First 3): {raw_logits[:3].cpu().numpy().flatten()}")
        print(f"    Probabilities (First 3): {probs[:3].cpu().numpy().flatten()}")

        assert (
            probs.min() >= 0.0 and probs.max() <= 1.0
        ), "Probabilities must be in [0, 1]"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
