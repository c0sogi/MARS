import os
import sys
import shutil
import warnings
import torch
import numpy as np
import pandas as pd

# Import library components
from library.config import Config
from library.utils import set_seed, get_device
from library.data import get_dataloaders
from library.model import DDARN
from library.loss import MaskedMCRMSELoss
from library.train import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== RNA Degradation Prediction: Library Demo ===\n")

    # 1. Configuration Patching for Demo
    # We modify the global Config to ensure the demo runs quickly and uses the working directory.
    print("[1] Configuring environment...")

    # Use a temporary directory for caching processed data
    demo_cache_dir = "./working/demo_cache"
    Config.CACHE_DIR = demo_cache_dir

    # Reduce training parameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 32

    # Set seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"    Device: {device}")
    print(f"    Cache Dir: {Config.CACHE_DIR}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Epochs: {Config.NUM_EPOCHS}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Pipeline...")

    # Load training data (load_cached=False forces processing from metadata CSVs)
    # This creates the .npz files in the demo_cache_dir
    train_loader = get_dataloaders(
        mode="train", load_cached=False, batch_size=Config.BATCH_SIZE
    )

    # Fetch a single batch
    X_batch, partners_batch, y_batch = next(iter(train_loader))

    # Move to device
    X_batch = X_batch.to(device)
    partners_batch = partners_batch.to(device)
    y_batch = y_batch.to(device)

    print(f"    Batch X shape: {X_batch.shape} (Expected: [B, 18, 107])")
    print(f"    Batch Partners shape: {partners_batch.shape} (Expected: [B, 107])")
    print(f"    Batch y shape: {y_batch.shape} (Expected: [B, 107, 5])")

    # Assertions
    assert X_batch.shape[1] == 18, "Input channels should be 18"
    assert X_batch.shape[2] == 107, "Sequence length should be 107"
    assert partners_batch.shape[1] == 107, "Partner length should be 107"
    assert y_batch.shape[2] == 5, "Target channels should be 5"

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture (DDARN)...")

    model = DDARN().to(device)

    # Pass 1: Zero Feedback (Initial Prediction)
    # The model expects prev_pred to be None or a tensor
    pred_1 = model(X_batch, partners_batch, prev_pred=None)

    print(f"    Pass 1 Output shape: {pred_1.shape} (Expected: [B, 5, 107])")
    assert pred_1.shape == (X_batch.shape[0], 5, 107), "Output shape mismatch"

    # Pass 2: Feedback Mechanism
    # We simulate the feedback loop used in training
    fb_input = pred_1.detach().clone()
    # Mask unscored positions (indices >= 68) as per training logic
    fb_input[:, :, Config.PRED_LEN :] = 0.0

    pred_2 = model(X_batch, partners_batch, prev_pred=fb_input)
    print(f"    Pass 2 Output shape: {pred_2.shape}")
    assert pred_2.shape == (
        X_batch.shape[0],
        5,
        107,
    ), "Feedback pass output shape mismatch"

    # 4. Loss Function Verification
    print("\n[4] Verifying Loss Function (MaskedMCRMSELoss)...")

    criterion = MaskedMCRMSELoss()

    # Calculate loss
    loss = criterion(pred_2, y_batch)
    print(f"    Calculated Loss: {loss.item():.6f}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Verify logic with identical inputs (Loss should be 0)
    # Note: We need to permute y_batch to match pred shape (B, 5, L) for this synthetic check
    # because the loss function handles (B, L, 5) internally, but if we pass identical tensors
    # we need to ensure they align before the loss function's internal permute logic or just rely on it.
    # The loss function expects: pred (B, 5, L), target (B, L, 5).
    # Let's create a perfect prediction tensor.
    perfect_pred = y_batch.permute(0, 2, 1).clone()  # (B, 5, L)
    perfect_loss = criterion(perfect_pred, y_batch)

    print(f"    Perfect Prediction Loss: {perfect_loss.item():.6f}")
    assert (
        perfect_loss.item() < 1e-6
    ), "Loss for perfect prediction should be effectively 0"

    # 5. Full Training Loop Execution
    print("\n[5] Executing Training Loop...")
    print("    Running for 2 epochs on the training set...")

    # run_training uses the Config settings we patched earlier
    # It will save 'best_model.pth' to Config.CACHE_DIR
    run_training(num_epochs=Config.NUM_EPOCHS, load_cached=True)

    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model file was not saved"
    print(f"    Training complete. Model saved to {best_model_path}")

    # 6. Inference Demonstration
    print("\n[6] Demonstrating Inference with Trained Model...")

    # Load validation data
    val_loader = get_dataloaders(
        mode="val", load_cached=True, batch_size=Config.BATCH_SIZE
    )

    # Load model weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    print("    Running inference on validation batch...")
    with torch.no_grad():
        X_val, p_val, y_val = next(iter(val_loader))
        X_val, p_val = X_val.to(device), p_val.to(device)

        # Inference typically involves the two-pass approach
        # Pass 1
        out_1 = model(X_val, p_val, prev_pred=None)

        # Prepare Feedback
        fb = out_1.clone()
        fb[:, :, Config.PRED_LEN :] = 0.0

        # Pass 2
        final_pred = model(X_val, p_val, prev_pred=fb)

    print(f"    Inference Output Shape: {final_pred.shape}")
    print("    Sample prediction (First 5 bases of first sample, Reactivity):")
    print(f"    {final_pred[0, 0, :5].cpu().numpy()}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
