import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import from the provided library
import library.config as config
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders, get_test_dataloader
from library.model import MSSHDNetwork
from library.trainer import Trainer
from library.inference import predict_and_submit


def run_demo():
    # 1. Setup & Configuration
    print(">>> Setting up demonstration...")
    seed_everything(42)
    device = get_device()

    # Define cache directory from config
    cache_dir = config.CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")

    # 2. Generate Mock Data to optimize speed
    # The real data processing takes too long for a demo.
    # We create synthetic cached .npy files so the data_loader skips raw DICOM processing.

    # Dimensions
    # Shape: (N, Channels, Height, Width) -> (N, 128, 224, 224)
    # Train needs at least BATCH_SIZE (32) because drop_last=True
    n_train = 32
    n_val = 10
    n_test = 10

    print(
        f">>> Generating mock data (Train: {n_train}, Val: {n_val}, Test: {n_test})..."
    )

    # Train Data
    X_train = np.random.rand(n_train, 128, 224, 224).astype(np.float32)
    y_train = np.random.randint(0, 2, size=(n_train,)).astype(np.float32)
    ids_train = np.array([f"{i:05d}" for i in range(n_train)])

    np.save(os.path.join(cache_dir, "cached_train_X.npy"), X_train)
    np.save(os.path.join(cache_dir, "cached_train_y.npy"), y_train)
    np.save(os.path.join(cache_dir, "cached_train_ids.npy"), ids_train)

    # Val Data
    X_val = np.random.rand(n_val, 128, 224, 224).astype(np.float32)
    y_val = np.random.randint(0, 2, size=(n_val,)).astype(np.float32)
    ids_val = np.array([f"{i+1000:05d}" for i in range(n_val)])

    np.save(os.path.join(cache_dir, "cached_val_X.npy"), X_val)
    np.save(os.path.join(cache_dir, "cached_val_y.npy"), y_val)
    np.save(os.path.join(cache_dir, "cached_val_ids.npy"), ids_val)

    # Test Data (No labels)
    X_test = np.random.rand(n_test, 128, 224, 224).astype(np.float32)
    ids_test = np.array([f"{i+2000:05d}" for i in range(n_test)])

    np.save(os.path.join(cache_dir, "cached_test_X.npy"), X_test)
    np.save(os.path.join(cache_dir, "cached_test_ids.npy"), ids_test)
    # y is usually not present for test, but logic handles it if missing

    print(">>> Mock data generated successfully.")

    # 3. Test Data Loading
    print("\n>>> Testing Data Loaders...")
    # load_cached_data=True ensures we pick up the files we just created
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    # Assertions
    assert (
        len(train_loader) > 0
    ), "Train loader is empty (check batch size vs dataset size)"
    assert len(val_loader) > 0, "Val loader is empty"

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))
    images = batch["image"]
    targets = batch["target"]

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    assert images.shape == (
        config.BATCH_SIZE,
        128,
        224,
        224,
    ), f"Incorrect image shape: {images.shape}"
    assert targets.shape == (
        config.BATCH_SIZE,
    ), f"Incorrect target shape: {targets.shape}"
    print("Data Loading Verified.")

    # 4. Test Model Architecture
    print("\n>>> Testing Model Architecture...")
    model = MSSHDNetwork().to(device)

    # Forward pass with the batch fetched earlier
    with torch.no_grad():
        logits = model(images.to(device))

    print(f"Model Output Shape: {logits.shape}")
    assert logits.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Incorrect output shape: {logits.shape}"
    print("Model Architecture Verified.")

    # 5. Test Training Loop
    print("\n>>> Testing Trainer...")
    # Initialize Trainer
    trainer = Trainer(train_loader, val_loader)

    # We want to run a very short training loop for the demo.
    # Since we cannot modify config.NUM_EPOCHS in the file, we rely on the loop running quickly
    # because our dataset is small (32 samples = 1 batch).
    # 15 epochs * 1 batch is very fast.

    trainer.run()

    # Verify checkpoint creation
    expected_model_path = os.path.join(cache_dir, "best_model.pth")
    assert os.path.exists(expected_model_path), "Best model checkpoint was not created."
    print(f"Training Verified. Model saved to {expected_model_path}")

    # 6. Test Inference and Submission
    print("\n>>> Testing Inference and Submission...")

    # Run inference
    # This will load the 'best_model.pth' we just trained and the mock test data
    predict_and_submit(load_cached_data=True)

    # Verify submission file
    submission_path = config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    # Check columns
    assert "BraTS21ID" in df_sub.columns, "Missing BraTS21ID column"
    assert "MGMT_value" in df_sub.columns, "Missing MGMT_value column"

    # Check row count (should match n_test)
    assert (
        len(df_sub) == n_test
    ), f"Submission row count {len(df_sub)} != Test set size {n_test}"

    # Check ID formatting (should be int as per requirement)
    assert pd.api.types.is_integer_dtype(
        df_sub["BraTS21ID"]
    ), "BraTS21ID should be integer"

    print("Inference Verified.")

    print("\n>>> DEMONSTRATION COMPLETE: All components functional.")


if __name__ == "__main__":
    run_demo()
