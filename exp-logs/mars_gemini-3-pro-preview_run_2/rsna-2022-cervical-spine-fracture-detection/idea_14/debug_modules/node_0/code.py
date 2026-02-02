import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_device, setup_logger
from library.data import get_dataloaders, get_test_dataloader
from library.model import CervicalFractureModel
from library.train_eval import WeightedMultilabelLoss, run_training


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Setup for Speed and Reproducibility
    # We modify the Config class attributes directly to suit a quick demo run.
    print("Configuring experiment parameters...")
    seed_everything(42)

    # Override Config values for the demonstration
    Config.DEBUG = True  # Uses a tiny subset of data (Batch Size * 2)
    Config.EXP_NAME = "demo_execution"
    Config.OUTPUT_DIR = os.path.join("./working", Config.EXP_NAME)
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.SEQ_LEN = 16  # Reduced from 96 to 16 for speed
    Config.IMAGE_SIZE = (256, 256)  # Reduced from 384 to 256
    Config.BACKBONE = "efficientnet_b0"  # Lighter backbone than b4
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo
    Config.GRAD_ACCUMULATION_STEPS = 1

    # Clean up previous run if exists to ensure a fresh start
    if os.path.exists(Config.OUTPUT_DIR):
        shutil.rmtree(Config.OUTPUT_DIR)
    Config.setup()

    device = get_device()
    print(f"Running on device: {device}")

    # 2. Demonstrate Data Loading
    print("\n[Step 1] Demonstrating Data Loading...")
    # load_cached_data=False forces scanning the directory (good for verifying logic)
    # With DEBUG=True, this will be very fast as it only processes a few items.
    train_loader, val_loader = get_dataloaders(load_cached_data=False)

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")

    # Fetch a single batch to verify shapes and types
    # The loader yields tuples of (images, targets)
    try:
        images, targets = next(iter(train_loader))
        print(f"Batch Images Shape: {images.shape}")  # Expected: (2, 16, 3, 256, 256)
        print(f"Batch Targets Shape: {targets.shape}")  # Expected: (2, 8)

        # Assertions to ensure data integrity
        expected_img_shape = (
            Config.BATCH_SIZE,
            Config.SEQ_LEN,
            3,
            Config.IMAGE_SIZE[0],
            Config.IMAGE_SIZE[1],
        )
        assert (
            images.shape == expected_img_shape
        ), f"Expected image shape {expected_img_shape}, got {images.shape}"
        assert targets.shape == (
            Config.BATCH_SIZE,
            8,
        ), f"Expected target shape {(Config.BATCH_SIZE, 8)}, got {targets.shape}"
        assert images.dtype == torch.float32, "Images should be float32"
        print("Data verification passed.")
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # 3. Model Instantiation and Forward Pass
    print("\n[Step 2] Demonstrating Model Architecture...")
    model = CervicalFractureModel()
    model.to(device)

    # Move batch to device for processing
    images = images.to(device)
    targets = targets.to(device)

    # Run forward pass
    with torch.no_grad():
        logits = model(images)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 8), "Logits shape mismatch"
    assert not torch.isnan(logits).any(), "Model produced NaN values"
    print("Model forward pass verification passed.")

    # 4. Loss Function Demonstration
    print("\n[Step 3] Demonstrating Loss Calculation...")
    criterion = WeightedMultilabelLoss(Config.LOSS_WEIGHTS, device)
    loss = criterion(logits, targets)

    print(f"Calculated Loss: {loss.item():.4f}")
    assert loss.dim() == 0, "Loss must be a scalar"
    assert loss.item() >= 0, "Loss must be non-negative"
    print("Loss function verification passed.")

    # 5. Full Training Loop Integration
    print("\n[Step 4] Running Training Loop (run_training)...")
    # run_training() encapsulates the entire training process:
    # - Re-initializes loaders (using our modified Config)
    # - Re-initializes model, optimizer, scheduler
    # - Runs training and validation loops
    # - Saves checkpoints
    run_training()

    # Verify artifact creation
    expected_checkpoint = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(expected_checkpoint):
        print(f"Checkpoint successfully saved at: {expected_checkpoint}")
    else:
        print(
            "Note: best_model.pth was not found. This is possible if validation loss did not improve in the single epoch."
        )

    # 6. Inference Demonstration
    print("\n[Step 5] Demonstrating Inference on Test Set...")
    test_loader = get_test_dataloader(load_cached_data=False)

    model.eval()
    print("Running inference on first 2 test studies...")

    with torch.no_grad():
        # Iterate through test loader
        # Note: Test dataset returns only images (no targets), so loader yields single tensor batches
        for i, batch_images in enumerate(test_loader):
            if i >= 2:
                break  # Limit to 2 samples

            batch_images = batch_images.to(device)
            logits = model(batch_images)
            probs = torch.sigmoid(logits)

            print(f"Sample {i} Predictions (Probabilities):")
            # Format output for display
            probs_np = probs.cpu().numpy()[0]
            for cls_name, prob in zip(Config.TARGET_COLS, probs_np):
                print(f"  {cls_name}: {prob:.4f}")

            assert probs.shape == (1, 8), "Prediction shape mismatch"

    print("\n=== Demonstration Script Completed Successfully ===")


if __name__ == "__main__":
    main()
