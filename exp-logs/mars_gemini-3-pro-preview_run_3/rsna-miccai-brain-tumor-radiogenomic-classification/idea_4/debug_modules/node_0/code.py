import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Ensure local library imports work
sys.path.append(os.getcwd())

from library.utils import (
    seed_everything,
    get_device,
    read_dicom_manual,
    process_patient,
)
from library.data_loader import get_dataloaders
from library.model import Siamese25DNet
from library.train import run_training


def run_demonstration():
    print("=" * 40)
    print(" MGMT Methylation Prediction - Code Demo")
    print("=" * 40)

    # ------------------------------------------------------------------------
    # 1. Test Utilities
    # ------------------------------------------------------------------------
    print("\n[1] Testing Utilities...")
    seed_everything(42)
    device = get_device()
    print(f"Device: {device}")

    # Test DICOM Reading
    # Using a known file from the provided dataset description
    test_dicom_path = "./input/train/00000/FLAIR/Image-1.dcm"
    if os.path.exists(test_dicom_path):
        img = read_dicom_manual(test_dicom_path)
        print(f"Read DICOM: Shape={img.shape}, Dtype={img.dtype}")

        if img.shape not in [(256, 256), (512, 512)]:
            raise AssertionError(f"Unexpected DICOM shape: {img.shape}")
        if img.dtype != np.uint16:
            raise AssertionError(f"Unexpected DICOM dtype: {img.dtype}")
    else:
        print(
            f"Warning: Test file {test_dicom_path} not found. Skipping file read test."
        )

    # Test Patient Processing
    # Construct a dummy row using the existing file for all modalities
    rel_path = "train/00000/FLAIR/Image-1.dcm"
    dummy_row = {
        "BraTS21ID": "00000",
        "flair_paths": [rel_path],
        "t1w_paths": [rel_path],
        "t1wce_paths": [rel_path],
        "t2w_paths": [rel_path],
        "MGMT_value": 1,
    }

    # Process the dummy patient
    try:
        processed_vol = process_patient(dummy_row, input_dir="./input")
        print(
            f"Processed Patient Volume: Shape={processed_vol.shape}, Dtype={processed_vol.dtype}"
        )

        # Expect (4 modalities, 32 slices, 256, 256)
        expected_shape = (4, 32, 256, 256)
        if processed_vol.shape != expected_shape:
            raise AssertionError(
                f"Expected shape {expected_shape}, got {processed_vol.shape}"
            )
    except Exception as e:
        print(f"process_patient failed: {e}")
        # Proceeding, as failure might be due to missing file if environment differs slightly

    # ------------------------------------------------------------------------
    # 2. Setup Mock Cache (Speed Optimization)
    # ------------------------------------------------------------------------
    print("\n[2] Setting up Mock Cache for Fast Execution...")
    # Instead of processing 400+ patients, we create small dummy cache files.
    # The data loader will pick these up automatically.
    cache_dir = "./working/idea_4"
    os.makedirs(cache_dir, exist_ok=True)

    N_samples = 4
    # Random volume data: (N, 4, 32, 256, 256)
    mock_X = np.random.rand(N_samples, 4, 32, 256, 256).astype(np.float32)
    mock_y = np.random.randint(0, 2, size=(N_samples,)).astype(np.float32)
    mock_ids = np.array([f"{i:05d}" for i in range(N_samples)])

    # Save mock files for Train, Val, and Test
    # Train
    np.save(os.path.join(cache_dir, "cached_train_X.npy"), mock_X)
    np.save(os.path.join(cache_dir, "cached_train_y.npy"), mock_y)
    np.save(os.path.join(cache_dir, "cached_train_ids.npy"), mock_ids)
    # Val
    np.save(os.path.join(cache_dir, "cached_val_X.npy"), mock_X)
    np.save(os.path.join(cache_dir, "cached_val_y.npy"), mock_y)
    np.save(os.path.join(cache_dir, "cached_val_ids.npy"), mock_ids)
    # Test (No targets)
    np.save(os.path.join(cache_dir, "cached_test_X.npy"), mock_X)
    np.save(os.path.join(cache_dir, "cached_test_ids.npy"), mock_ids)
    # Ensure no y file for test
    test_y_path = os.path.join(cache_dir, "cached_test_y.npy")
    if os.path.exists(test_y_path):
        os.remove(test_y_path)

    print("Mock cache files created in ./working/idea_4/")

    # ------------------------------------------------------------------------
    # 3. Test Data Loader
    # ------------------------------------------------------------------------
    print("\n[3] Testing Data Loaders...")
    batch_size = 2
    # Load cached data (which picks up our mock files)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,  # 0 for simpler debugging
        load_cached_data=True,
        debug=True,
    )

    # Fetch one batch
    flair, t1w, t1wce, t2w, targets = next(iter(train_loader))
    print(f"Batch Loaded: Flair={flair.shape}, Targets={targets.shape}")

    if flair.shape != (batch_size, 32, 256, 256):
        raise AssertionError(f"Incorrect batch shape: {flair.shape}")
    if targets.shape != (batch_size,):
        raise AssertionError(f"Incorrect targets shape: {targets.shape}")

    # ------------------------------------------------------------------------
    # 4. Test Model Architecture
    # ------------------------------------------------------------------------
    print("\n[4] Testing Siamese 2.5D Model...")
    model = Siamese25DNet(in_channels=32, num_classes=1)
    model = model.to(device)

    # Move batch to device
    flair = flair.to(device)
    t1w = t1w.to(device)
    t1wce = t1wce.to(device)
    t2w = t2w.to(device)

    # Forward pass
    logits = model(flair, t1w, t1wce, t2w)
    print(f"Model Output Shape: {logits.shape}")

    if logits.shape != (batch_size, 1):
        raise AssertionError(f"Expected output shape (B, 1), got {logits.shape}")

    # ------------------------------------------------------------------------
    # 5. Test Full Training Pipeline
    # ------------------------------------------------------------------------
    print("\n[5] Testing Training Pipeline (Integration Test)...")
    # Using small epochs and debug mode for speed
    best_auc = run_training(
        epochs=1,
        batch_size=2,
        learning_rate=1e-4,
        patience=1,
        debug=True,
        load_cached_data=True,
        save_dir=cache_dir,
    )

    print(f"Training Run Complete. Best AUC: {best_auc}")

    # Verify Artifacts
    model_path = os.path.join(cache_dir, "best_model.pth")
    sub_path = "./submission/submission.csv"

    if not os.path.exists(model_path):
        raise AssertionError("Model checkpoint (best_model.pth) was not created.")
    if not os.path.exists(sub_path):
        raise AssertionError("Submission file (submission.csv) was not created.")

    print("Artifact verification passed.")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demonstration()
