import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything
from library.model import NFWBN
from library.data_loader import get_loaders
from library.train_eval import train_one_epoch, validate


def run_demo():
    print("=== Starting Demonstration of Iceberg Classification Solution ===\n")

    # 1. Setup and Configuration overrides for speed
    print("[1] Setting up configuration and seeding...")
    seed_everything(Config.SEED)

    # Override Config for rapid demonstration
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Small subset for speed
    Config.NUM_EPOCHS = 2  # Only run 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.WORK_DIR = "./working/demo_execution"
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"    Epochs: {Config.NUM_EPOCHS}")
    print(f"    Device: {Config.DEVICE}")

    # 2. Model Architecture Verification
    print("\n[2] Verifying NFWBN Model Architecture...")
    model = NFWBN().to(Config.DEVICE)

    # Create dummy inputs
    # Image: (Batch, 3, 75, 75) - 3 channels as per IcebergDataset (Band1, Band2, Avg)
    dummy_img = torch.randn(4, 3, 75, 75).to(Config.DEVICE)
    # Angle: (Batch, 1)
    dummy_angle = torch.randn(4, 1).to(Config.DEVICE)

    # Forward pass
    try:
        output = model(dummy_img, dummy_angle)
        print("    Forward pass successful.")
        print(f"    Input shapes: Img {dummy_img.shape}, Angle {dummy_angle.shape}")
        print(f"    Output shape: {output.shape}")

        # Assertions
        assert output.shape == (
            4,
            1,
        ), f"Expected output shape (4, 1), got {output.shape}"
        assert not torch.isnan(output).any(), "Model output contains NaNs"
        print("    Model architecture verification passed.")
    except Exception as e:
        print(f"    Model verification failed: {e}")
        raise e

    # 3. Data Loading Verification
    print("\n[3] Verifying Data Loaders...")
    try:
        # get_loaders handles stats calculation and dataset creation
        train_loader, val_loader, test_loader = get_loaders(
            batch_size=Config.BATCH_SIZE,
            num_workers=0,  # Use 0 workers for simple script execution to avoid multiprocessing overhead
            debug=Config.DEBUG,
        )

        print("    DataLoaders created successfully.")

        # Fetch one batch from train_loader
        imgs, angles, targets = next(iter(train_loader))

        print(
            f"    Train Batch - Images: {imgs.shape}, Angles: {angles.shape}, Targets: {targets.shape}"
        )

        # Validate shapes
        # Batch size might be smaller if subset size is not divisible, but here 32 % 8 == 0
        assert imgs.shape[0] == Config.BATCH_SIZE
        assert imgs.shape[1] == 3  # 3 Channels
        assert imgs.shape[2] == 75  # Height
        assert imgs.shape[3] == 75  # Width
        assert angles.shape[1] == 1  # 1 Angle feature
        assert targets.shape[1] == 1  # 1 Target

        print("    Data Loader verification passed.")

    except Exception as e:
        print(f"    Data Loader verification failed: {e}")
        raise e

    # 4. Training Loop Verification
    print("\n[4] Verifying Training and Validation Loop...")

    # Setup optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    initial_weights = list(model.parameters())[0].clone()

    try:
        for epoch in range(Config.NUM_EPOCHS):
            print(f"    Running Epoch {epoch + 1}/{Config.NUM_EPOCHS}...")

            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, Config.DEVICE
            )

            # Validate
            val_loss = validate(model, val_loader, criterion, Config.DEVICE)

            print(f"        Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # Assertions
            assert not np.isnan(train_loss), "Training loss is NaN"
            assert not np.isnan(val_loss), "Validation loss is NaN"
            assert train_loss >= 0, "Training loss should be non-negative"
            assert val_loss >= 0, "Validation loss should be non-negative"

        # Check if weights updated
        final_weights = list(model.parameters())[0]
        assert not torch.equal(
            initial_weights, final_weights
        ), "Model weights did not update during training!"
        print("    Training loop verification passed.")

    except Exception as e:
        print(f"    Training loop failed: {e}")
        raise e

    # 5. Inference Verification
    print("\n[5] Verifying Inference on Test Set...")
    try:
        model.eval()
        test_imgs, test_angles, test_ids = next(iter(test_loader))

        test_imgs = test_imgs.to(Config.DEVICE)
        test_angles = test_angles.to(Config.DEVICE)

        with torch.no_grad():
            logits = model(test_imgs, test_angles)
            probs = torch.sigmoid(logits)

        print(f"    Test Batch Size: {len(test_ids)}")
        print(f"    Predictions Shape: {probs.shape}")
        print(f"    Sample Predictions: {probs.flatten()[:5].cpu().numpy()}")

        assert probs.shape == (len(test_ids), 1)
        assert (probs >= 0).all() and (
            probs <= 1
        ).all(), "Probabilities must be between 0 and 1"

        print("    Inference verification passed.")

    except Exception as e:
        print(f"    Inference verification failed: {e}")
        raise e

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
