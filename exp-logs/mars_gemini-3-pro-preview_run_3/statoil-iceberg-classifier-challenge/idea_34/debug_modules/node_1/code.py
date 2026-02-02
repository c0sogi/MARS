import os
import sys
import torch
import numpy as np
import pandas as pd

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_data_loaders, load_data
from library.model import SDHAResNet
from library.trainer import train_fold


def main():
    print("=== Iceberg Classifier Demo Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Enable Debug mode to use a small subset of data for speed
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 100  # Use 100 samples for train/test

    # Reduce training parameters for quick execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_FOLDS = 2  # We will only run Fold 0

    # Set a specific working directory for this demo run
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Redirect Cache Paths to the existing cache to save processing time
    # The environment info indicates cache is at ./working/cache
    EXISTING_CACHE_DIR = "./working/cache"
    for key in Config.CACHE_PATHS:
        # Map the config keys to the expected filenames in the cache dir
        # Note: The provided dataset.py expects specific keys.
        # We assume the filenames match the keys + .npy extension.
        Config.CACHE_PATHS[key] = os.path.join(EXISTING_CACHE_DIR, f"{key}.npy")

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Cache Directory: {EXISTING_CACHE_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    device = get_device()
    print(f"    Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Load data loaders for Fold 0
    # This will load data from cache (if available) and subset it because DEBUG=True
    train_loader, val_loader, test_loader = get_data_loaders(
        fold_idx=0, load_cached_data=True
    )

    # Fetch one batch to verify shapes
    try:
        images, angles, labels = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    print(
        f"    Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Assertions to ensure data integrity
    expected_img_shape = (Config.BATCH_SIZE, 3, 75, 75)
    expected_lbl_shape = (Config.BATCH_SIZE,)

    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
    assert (
        angles.shape == expected_lbl_shape
    ), f"Angle shape mismatch. Expected {expected_lbl_shape}, got {angles.shape}"
    assert (
        labels.shape == expected_lbl_shape
    ), f"Label shape mismatch. Expected {expected_lbl_shape}, got {labels.shape}"

    print("    Data Pipeline verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = SDHAResNet().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Perform forward pass
    with torch.no_grad():
        logits = model(images, angles)

    print(f"    Output Logits Shape: {logits.shape}")

    # Assert output shape
    assert (
        logits.shape == expected_lbl_shape
    ), f"Model output shape mismatch. Expected {expected_lbl_shape}, got {logits.shape}"

    print("    Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (Fold 0)...")

    # Run training for Fold 0
    # This uses the trainer.py logic which includes training, validation, and checkpointing
    best_val_loss = train_fold(fold_idx=0, load_cached_data=True)

    print(f"    Training finished. Best Validation Log Loss: {best_val_loss:.4f}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "model_fold_0.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"    Checkpoint verified at: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5] Running Inference on Test Set...")

    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    predictions = []

    # Inference loop
    with torch.no_grad():
        for images, angles in test_loader:
            images = images.to(device)
            angles = angles.to(device)

            logits = model(images, angles)
            probs = torch.sigmoid(logits)

            predictions.extend(probs.cpu().numpy())

    predictions = np.array(predictions)
    print(f"    Generated {len(predictions)} predictions.")

    # Generate Submission CSV
    # Note: In DEBUG mode, the test set is a random subset.
    # For this demo, we will create a valid CSV format using the first N IDs from the cache.
    # In a real run (DEBUG=False), the test_loader order matches ids_test exactly.

    # Load IDs from cache to create the dataframe
    ids_test = np.load(Config.CACHE_PATHS["ids_test"])

    # Handle the size mismatch caused by DEBUG subsetting
    if len(predictions) != len(ids_test):
        print(
            f"    Debug mode active: Using first {len(predictions)} IDs for demo submission."
        )
        ids_test = ids_test[: len(predictions)]

    submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": predictions})

    submission_path = "demo_submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"    Submission saved to: {submission_path}")
    print(f"    First 3 rows:\n{submission_df.head(3)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
