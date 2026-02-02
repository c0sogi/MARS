import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.utils import set_seed, get_device
from library.data_loader import get_loaders
from library.model import IcebergCNN
from library.train import run_fold
from library.predict import generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Reproducibility
    print("\n[Step 1] Setting random seed...")
    set_seed(42)
    device = get_device()
    print(f"Device selected: {device}")

    # 2. Data Loading
    # We use a small batch size for the demo.
    # Note: The first run might take a few seconds to process JSONs into NumPy arrays
    # and cache them in ./working/idea_30/.
    print("\n[Step 2] Loading DataLoaders...")
    batch_size = 8
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=batch_size, num_workers=2
    )

    # Verification: Check batch structure and shapes
    print("Verifying DataLoader outputs...")
    # Fetch one batch from training loader
    (imgs, angles), labels = next(iter(train_loader))

    # Expected shapes:
    # Images: (batch_size, 3, 75, 75) - 3 channels (HH, HV, Avg)
    # Angles: (batch_size,)
    # Labels: (batch_size,)
    assert imgs.shape == (
        batch_size,
        3,
        75,
        75,
    ), f"Unexpected image shape: {imgs.shape}"
    assert angles.shape == (batch_size,), f"Unexpected angle shape: {angles.shape}"
    assert labels.shape == (batch_size,), f"Unexpected label shape: {labels.shape}"

    print(
        f"Batch shapes verified: Imgs {imgs.shape}, Angles {angles.shape}, Labels {labels.shape}"
    )

    # 3. Model Instantiation and Forward Pass
    print("\n[Step 3] Instantiating IcebergCNN model...")
    model = IcebergCNN().to(device)

    # Verification: Run a dummy forward pass
    print("Verifying model forward pass...")
    imgs_dev = imgs.to(device)
    angles_dev = angles.to(device)

    with torch.no_grad():
        output = model(imgs_dev, angles_dev)

    # Expected output shape: (batch_size, 1) -> Logits
    assert output.shape == (batch_size, 1), f"Unexpected output shape: {output.shape}"
    print(f"Forward pass successful. Output shape: {output.shape}")

    # 4. Training Demonstration
    # We will train for just 1 epoch to demonstrate functionality quickly.
    # We use a custom checkpoint directory for this demo.
    print("\n[Step 4] Running Training Loop (Fold 0, 1 Epoch)...")
    demo_checkpoint_dir = "./working/demo_checkpoints"

    # Clean up previous demo checkpoints if they exist
    if os.path.exists(demo_checkpoint_dir):
        shutil.rmtree(demo_checkpoint_dir)

    trained_model = run_fold(
        train_loader=train_loader,
        val_loader=val_loader,
        fold_idx=0,
        epochs=1,  # Reduced for speed
        patience=1,
        lr=1e-3,
        save_dir=demo_checkpoint_dir,
    )

    # Verification: Check if checkpoint file was created
    expected_checkpoint = os.path.join(demo_checkpoint_dir, "model_best_fold_0.pth")
    if not os.path.exists(expected_checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {expected_checkpoint}")
    print(f"Training complete. Checkpoint saved at {expected_checkpoint}")

    # 5. Prediction and Submission Generation
    print("\n[Step 5] Generating Submission...")
    demo_submission_path = "./working/demo_submission.csv"

    # We only trained fold 0, so we set num_folds=1
    generate_submission(
        test_loader=test_loader,
        checkpoint_dir=demo_checkpoint_dir,
        output_path=demo_submission_path,
        num_folds=1,
    )

    # Verification: Check submission file format
    if not os.path.exists(demo_submission_path):
        raise FileNotFoundError(f"Submission file not found at {demo_submission_path}")

    df_sub = pd.read_csv(demo_submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Check columns
    expected_cols = ["id", "is_iceberg"]
    if list(df_sub.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        )

    # Check row count (should match test set size = 321)
    # Note: test.json has 321 lines in the description provided, but let's verify against the loader dataset length
    expected_len = len(test_loader.dataset)
    if len(df_sub) != expected_len:
        raise ValueError(
            f"Submission length mismatch. Expected {expected_len}, got {len(df_sub)}"
        )

    # Check probability range
    if df_sub["is_iceberg"].min() < 0 or df_sub["is_iceberg"].max() > 1:
        raise ValueError("Probabilities out of range [0, 1]")

    print("Submission format verified successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
