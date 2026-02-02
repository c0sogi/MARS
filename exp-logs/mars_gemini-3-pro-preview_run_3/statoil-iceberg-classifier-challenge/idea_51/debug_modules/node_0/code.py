import os
import sys
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.dataset import get_loaders
from library.model import DPDCNN
from library.train import train_one_fold
from library.utils import log_loss_score, set_seed


def main():
    print("=== Starting Demonstration of Iceberg Classification Pipeline ===")

    # 1. Setup and Configuration
    # Initialize directories and seeds
    Config.setup()

    # Enable Debug mode to use a small subset of data and fewer epochs for speed
    print("\n[Step 1] Configuring Debug Mode...")
    Config.set_debug_mode(debug=True, max_samples=64, epochs=2)

    # Ensure reproducibility
    set_seed(Config.SEED)

    # 2. Data Loading
    print("\n[Step 2] Loading Data...")
    # get_loaders handles reading metadata, processing raw JSON (or loading cache),
    # and creating PyTorch DataLoaders.
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Verify Train Loader
    try:
        images, angles, targets = next(iter(train_loader))
        print(f"  Train Batch - Images Shape: {images.shape}")
        print(f"  Train Batch - Angles Shape: {angles.shape}")
        print(f"  Train Batch - Targets Shape: {targets.shape}")

        # Assertions to ensure data pipeline is correct
        assert images.dim() == 4, "Images must be 4D tensors (B, C, H, W)"
        assert images.size(1) == 3, "Images must have 3 channels (HH, HV, Avg)"
        assert images.size(2) == 75 and images.size(3) == 75, "Images must be 75x75"
        assert angles.dim() == 1, "Angles must be 1D tensors"
        assert targets.dim() == 1, "Targets must be 1D tensors"
        print("  Data shapes verified successfully.")
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # 3. Model Instantiation and Forward Pass
    print("\n[Step 3] Initializing Model and Testing Forward Pass...")
    device = torch.device(Config.DEVICE)
    model = DPDCNN().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(images, angles)

    print(f"  Model Output Shape: {outputs.shape}")

    # Assertions for model output
    assert outputs.dim() == 2, "Model output should be 2D (B, 1)"
    assert outputs.size(1) == 1, "Model output size should be 1 (logit)"
    assert outputs.size(0) == images.size(0), "Batch size mismatch in output"
    print("  Model forward pass verified successfully.")

    # 4. Training Loop Demonstration
    print("\n[Step 4] Running Training Loop for Fold 0 (Debug Mode)...")
    # train_one_fold handles the optimizer, loss, backprop, and validation logic
    best_state, best_preds, best_targets = train_one_fold(0, train_loader, val_loader)

    # Verify training outputs
    print(f"  Returned Predictions Shape: {best_preds.shape}")
    print(f"  Returned Targets Shape: {best_targets.shape}")

    assert len(best_preds) == len(
        best_targets
    ), "Mismatch between preds and targets length"
    assert len(best_preds) > 0, "No predictions returned"

    # Check if predictions are probabilities (0-1) as expected from train_one_fold
    assert np.all(
        (best_preds >= 0) & (best_preds <= 1)
    ), "Predictions must be probabilities [0, 1]"
    print("  Training loop execution verified successfully.")

    # 5. Metric Calculation
    print("\n[Step 5] Verifying Metric Calculation...")
    # Calculate log loss on the validation set returned by the training loop
    loss = log_loss_score(best_targets, best_preds)
    print(f"  Calculated Validation Log Loss: {loss:.4f}")

    # Sanity check: Loss should be non-negative
    assert loss >= 0, "Log loss cannot be negative"

    # 6. Inference / Checkpoint Loading Simulation
    print("\n[Step 6] Simulating Inference with Saved Checkpoint...")

    # Path to the saved checkpoint
    checkpoint_path = os.path.join(Config.WORKING_DIR, "model_best_fold_0.pth")
    if not os.path.exists(checkpoint_path):
        # Fallback if best model wasn't saved (e.g. if loss didn't improve, though unlikely in 2 epochs starting from scratch)
        checkpoint_path = os.path.join(Config.WORKING_DIR, "checkpoint_fold_0.pth")

    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Run inference on a test batch
    try:
        test_images, test_angles, test_ids = next(iter(test_loader))
        test_images = test_images.to(device)
        test_angles = test_angles.to(device)

        with torch.no_grad():
            test_logits = model(test_images, test_angles)
            test_probs = torch.sigmoid(test_logits).cpu().numpy().ravel()

        print(f"  Test Batch IDs: {test_ids[:3]}...")
        print(f"  Test Predictions: {test_probs[:3]}...")

        assert len(test_probs) == test_images.size(0), "Test output size mismatch"
        print("  Inference simulation verified successfully.")

    except StopIteration:
        print(
            "  Test loader is empty (expected if debug sample size is very small), skipping inference check."
        )

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
