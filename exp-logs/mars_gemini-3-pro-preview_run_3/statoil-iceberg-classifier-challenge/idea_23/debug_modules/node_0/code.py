import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.data_loader import get_loaders
from library.model import MSMANet
from library.train_eval import train_fold


def main():
    print("=== Starting Demo for Iceberg Detection Solution ===\n")

    # 1. Setup Configuration for Fast Demonstration
    print("Step 1: Configuring environment for fast execution...")
    Config.WORKING_DIR = "./working/demo_run"
    Config.DEBUG = True  # Use subset of data
    Config.MAX_DEBUG_SAMPLES = 100  # Only 100 samples
    Config.EPOCHS = 2  # Only 2 epochs
    Config.BATCH_SIZE = 8
    Config.NUM_FOLDS = 2  # We will only run fold 0
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=2, BATCH_SIZE=8")

    # 2. Data Loader Verification
    print("\nStep 2: Verifying Data Loaders...")
    train_loader, val_loader, test_loader = get_loaders(fold_idx=0, debug=Config.DEBUG)

    # Fetch one batch
    images, angles, labels = next(iter(train_loader))

    print(f"Train Batch - Images Shape: {images.shape}")
    print(f"Train Batch - Angles Shape: {angles.shape}")
    print(f"Train Batch - Labels Shape: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), "Incorrect image tensor shape"
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect angle tensor shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label tensor shape"
    assert images.dtype == torch.float32, "Images should be float32"
    print("Data Loader verification passed.")

    # 3. Model Instantiation and Forward Pass
    print("\nStep 3: Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = MSMANet().to(device)

    # Move batch to device
    images_dev = images.to(device)
    angles_dev = angles.to(device)

    # Forward pass
    logits = model(images_dev, angles_dev)

    print(f"Model Output Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (Config.BATCH_SIZE, 1), "Output shape should be (B, 1)"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"
    print("Model architecture verification passed.")

    # 4. Full Training Loop Execution
    print("\nStep 4: Executing Training Loop (Fold 0)...")
    # This calls the library function which handles the loop, validation, and saving
    best_val_loss = train_fold(fold_idx=0)

    print(f"Training completed. Best Validation Loss: {best_val_loss:.4f}")
    assert isinstance(best_val_loss, float), "train_fold should return a float loss"

    # 5. Artifact Verification
    print("\nStep 5: Verifying Output Artifacts...")
    checkpoint_path = os.path.join(Config.WORKING_DIR, "model_best_fold_0.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"Checkpoint successfully created at: {checkpoint_path}")

    # 6. Inference Simulation
    print("\nStep 6: Simulating Inference on Test Data...")

    # Load the best model
    inference_model = MSMANet().to(device)
    checkpoint = load_checkpoint(checkpoint_path, inference_model, device=Config.DEVICE)
    inference_model.eval()

    print(
        f"Loaded model from epoch {checkpoint['epoch']} with loss {checkpoint['best_val_loss']:.4f}"
    )

    # Get a test batch
    test_images, test_angles, test_ids = next(iter(test_loader))
    test_images = test_images.to(device)
    test_angles = test_angles.to(device)

    # Predict
    with torch.no_grad():
        test_logits = inference_model(test_images, test_angles)
        test_probs = torch.sigmoid(test_logits)

    # Display sample predictions
    print("\nSample Predictions:")
    print(f"{'ID':<15} | {'Probability (Iceberg)':<20}")
    print("-" * 40)
    for i in range(min(5, len(test_ids))):
        print(f"{test_ids[i]:<15} | {test_probs[i].item():.4f}")

    # Final assertion on probability range
    assert (
        test_probs.min() >= 0.0 and test_probs.max() <= 1.0
    ), "Probabilities must be in [0, 1]"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
