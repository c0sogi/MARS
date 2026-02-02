import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path for library imports
sys.path.append(os.getcwd())

from library.utils import set_seed
from library.dataset import get_dataloaders
from library.network import IcebergResNet
from library.trainer import train_model, predict_and_submit
from library.config import MODEL_SAVE_PATH, SUBMISSION_DIR, IMAGE_SIZE


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup
    print("\n[Step 1] Setting random seed...")
    set_seed(42)

    # 2. Data Loading
    print("\n[Step 2] Loading DataLoaders...")
    # Using a small batch size for demonstration purposes
    batch_size = 8
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        batch_size=batch_size, num_workers=2, load_cached_data=True
    )

    # Verify Train Loader
    print("Verifying Train Loader batch structure...")
    images, angles, labels = next(iter(train_loader))

    # Assertions for data shapes
    # Image: (B, 3, 224, 224)
    assert images.shape == (
        batch_size,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
    ), f"Expected image shape {(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)}, got {images.shape}"
    # Angle: (B, 1)
    assert angles.shape == (
        batch_size,
        1,
    ), f"Expected angle shape {(batch_size, 1)}, got {angles.shape}"
    # Label: (B, 1)
    assert labels.shape == (
        batch_size,
        1,
    ), f"Expected label shape {(batch_size, 1)}, got {labels.shape}"

    print("Data Loader verification successful.")

    # 3. Model Initialization & Forward Pass
    print("\n[Step 3] Initializing Model and checking forward pass...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = IcebergResNet().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(images, angles)

    # Verify output shape and range
    assert outputs.shape == (
        batch_size,
        1,
    ), f"Expected output shape {(batch_size, 1)}, got {outputs.shape}"
    assert (
        outputs.min() >= 0 and outputs.max() <= 1
    ), "Model outputs should be probabilities between 0 and 1 (Sigmoid)."

    print("Model forward pass verification successful.")

    # 4. Training
    print("\n[Step 4] Running Training Loop (1 Epoch)...")
    # We run for just 1 epoch to demonstrate functionality quickly
    trained_model = train_model(train_loader, val_loader, num_epochs=1, patience=1)

    # Verify model checkpoint creation
    assert os.path.exists(
        MODEL_SAVE_PATH
    ), f"Model file was not saved at {MODEL_SAVE_PATH}"
    print(f"Training complete. Model saved at {MODEL_SAVE_PATH}")

    # 5. Inference
    print("\n[Step 5] Generating Submission...")
    demo_submission_path = os.path.join(SUBMISSION_DIR, "demo_submission.csv")

    predict_and_submit(
        trained_model, test_loader, test_ids, output_path=demo_submission_path
    )

    # Verify submission file
    assert os.path.exists(demo_submission_path), "Submission file was not created."

    df_sub = pd.read_csv(demo_submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Check columns
    assert list(df_sub.columns) == [
        "id",
        "is_iceberg",
    ], f"Incorrect columns: {list(df_sub.columns)}"

    # Check row count (Test set size is 321 based on description)
    expected_rows = 321
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check value range
    assert (
        df_sub["is_iceberg"].min() >= 0 and df_sub["is_iceberg"].max() <= 1
    ), "Submission probabilities out of range [0, 1]."

    print("Submission verification successful.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
