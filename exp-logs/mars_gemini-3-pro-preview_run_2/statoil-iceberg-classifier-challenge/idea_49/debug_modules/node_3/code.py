import os
import sys
import torch
import numpy as np
import logging

# Add the current directory to sys.path to ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import setup_logger
from library.data_loader import get_dataloaders, get_test_dataloader
from library.model import DCSWBN
from library.train_eval import train_fold


def run_demo():
    # ==========================================
    # 1. Configuration for Demo
    # ==========================================
    print("1. Configuring environment for demo...")

    # Modify Config for a quick run
    Config.DEBUG = True
    Config.DEBUG_SIZE = 50  # Small subset for speed
    Config.NUM_EPOCHS = 2  # Only 2 epochs to verify loop
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Setup Logger
    log_path = os.path.join(Config.WORK_DIR, "demo.log")
    logger = setup_logger(log_path, name="demo_logger")
    logger.info("Demo configuration applied.")

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    logger.info("\n2. Verifying Data Loading...")

    # Get dataloaders for Fold 0
    # This triggers data processing (if not cached) and loading
    train_loader, val_loader = get_dataloaders(
        fold_index=0, load_cached_data=True, debug=True
    )

    # Fetch one batch from training loader
    images, angles, labels, ids = next(iter(train_loader))

    # Verify Shapes
    logger.info(
        f"Batch shapes - Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions
    assert images.dim() == 4, "Images should be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert images.shape[2] == 75 and images.shape[3] == 75, "Images should be 75x75"
    assert angles.dim() == 1, "Angles should be 1D tensors"
    assert labels.dim() == 1, "Labels should be 1D tensors"
    assert len(ids) == images.shape[0], "Number of IDs should match batch size"

    logger.info("Data loader verification passed.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    logger.info("\n3. Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = DCSWBN().to(device)

    # Move batch to device
    images_dev = images.to(device)
    angles_dev = angles.to(device)

    # Forward pass
    outputs = model(images_dev, angles_dev)

    logger.info(f"Model output shape: {outputs.shape}")

    # Assertions
    assert outputs.dim() == 2, "Output should be 2D (B, 1)"
    assert outputs.shape[0] == images.shape[0], "Output batch size should match input"
    assert outputs.shape[1] == 1, "Output channel should be 1 (binary logits)"

    logger.info("Model architecture verification passed.")

    # ==========================================
    # 4. Training Pipeline Demonstration
    # ==========================================
    logger.info("\n4. Running Training Fold Demo...")

    # Run training for one fold (shortened by Config modifications)
    # train_fold returns the best model state dict
    best_state_dict = train_fold(
        fold_index=0, train_loader=train_loader, val_loader=val_loader, logger=logger
    )

    assert best_state_dict is not None, "train_fold should return a state dict"
    logger.info("Training fold execution passed.")

    # ==========================================
    # 5. Inference Demonstration
    # ==========================================
    logger.info("\n5. Verifying Inference...")

    # Load test loader
    test_loader = get_test_dataloader(load_cached_data=True)

    # Load best model
    model.load_state_dict(best_state_dict)
    model.eval()

    # Get a test batch
    test_images, test_angles, test_ids = next(iter(test_loader))
    test_images = test_images.to(device)
    test_angles = test_angles.to(device)

    with torch.no_grad():
        test_logits = model(test_images, test_angles)
        test_probs = torch.sigmoid(test_logits)

    logger.info(
        f"Test batch predictions (first 3): {test_probs[:3].cpu().numpy().flatten()}"
    )

    # Assertions
    assert (
        test_probs.min() >= 0.0 and test_probs.max() <= 1.0
    ), "Probabilities must be in [0, 1]"
    assert len(test_probs) == len(
        test_ids
    ), "Number of predictions must match number of IDs"

    logger.info("Inference verification passed.")
    logger.info("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
