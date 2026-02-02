import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config, load_and_process_data, IcebergDataset
from library.model import MDSWBN
from library.train import train_one_epoch, validate, run_training
from library.utils import seed_everything
from library.data_loader import get_loaders, get_test_loader


def run_demo():
    print("Initializing Demo...")

    # 1. Setup Configuration for Demo
    # We modify the Config class attributes directly to isolate demo artifacts
    # and speed up execution.
    Config.WORKING_DIR = "./working/demo_artifacts"
    Config.SUBMISSION_PATH = "./working/demo_submission.csv"
    Config.N_FOLDS = 2  # Reduce folds for speed
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(42)

    print("\n=== Step 1: Data Loading and Processing Verification ===")
    # Load a small subset of data (limit_data=20)
    # This tests load_and_process_data from library.config
    X, y, inc, X_test, inc_test, test_ids = load_and_process_data(
        load_cached_data=False, limit_data=20
    )

    print(f"Loaded Training Data Shape: {X.shape}")
    print(f"Loaded Target Data Shape: {y.shape}")
    print(f"Loaded Incidence Angle Shape: {inc.shape}")

    # Assertions to verify data integrity
    assert X.shape == (
        20,
        3,
        75,
        75,
    ), f"Expected X shape (20, 3, 75, 75), got {X.shape}"
    assert y.shape == (20,), f"Expected y shape (20,), got {y.shape}"
    assert inc.shape == (20,), f"Expected inc shape (20,), got {inc.shape}"
    assert not np.isnan(X).any(), "Input data contains NaNs"

    print("Data loading verification passed.")

    print("\n=== Step 2: Dataset and DataLoader Verification ===")
    # Instantiate the Dataset class
    dataset = IcebergDataset(X, inc, y, transform=True)

    # Check single item retrieval
    img, angle, label = dataset[0]
    print(f"Single Item - Image Shape: {img.shape}, Angle: {angle}, Label: {label}")

    assert img.shape == (3, 75, 75), "Incorrect image tensor shape"
    assert isinstance(angle, torch.Tensor), "Angle should be a tensor"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"

    # Check DataLoader
    train_loader, val_loader = get_loaders(
        fold_idx=0,
        X=X,
        y=y,
        inc_angles=inc,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
    )

    batch_imgs, batch_angles, batch_labels = next(iter(train_loader))
    print(
        f"Batch Shapes - Images: {batch_imgs.shape}, Angles: {batch_angles.shape}, Labels: {batch_labels.shape}"
    )

    assert batch_imgs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_imgs.shape[1:] == (3, 75, 75), "Image dimensions mismatch in batch"

    print("Dataset and DataLoader verification passed.")

    print("\n=== Step 3: Model Architecture Verification ===")
    # Instantiate the MDSWBN model
    model = MDSWBN().to(Config.DEVICE)

    # Move batch to device
    batch_imgs = batch_imgs.to(Config.DEVICE)
    batch_angles = batch_angles.to(Config.DEVICE)

    # Forward pass
    output = model(batch_imgs, batch_angles)
    print(f"Model Output Shape: {output.shape}")
    print(f"Model Output Values: {output.detach().cpu().numpy().flatten()}")

    # Assertions
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert (output >= 0).all() and (
        output <= 1
    ).all(), "Model output not in [0, 1] range (Sigmoid check)"

    print("Model architecture verification passed.")

    print("\n=== Step 4: Training Loop Component Verification ===")
    # Setup for single epoch training
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # Train for one epoch on the small subset
    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion, Config.DEVICE
    )
    print(f"Train Loss (1 epoch): {train_loss:.6f}")

    # Validate
    val_loss = validate(model, val_loader, criterion, Config.DEVICE)
    print(f"Validation Loss: {val_loss:.6f}")

    assert not np.isnan(train_loss), "Training loss is NaN"
    assert not np.isnan(val_loss), "Validation loss is NaN"

    print("Training loop components verification passed.")

    print("\n=== Step 5: Full Pipeline Execution Verification ===")
    print("Running full training pipeline (debug mode, 1 epoch)...")

    # We use the library's run_training function which orchestrates the K-Fold CV
    # We set debug=True to limit data size inside the function as well
    # We set epochs=1 to ensure it finishes quickly
    run_training(epochs=1, debug=True)

    # Verify submission file creation
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file successfully created at {Config.SUBMISSION_PATH}")
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file shape: {df_sub.shape}")
        print(df_sub.head())

        # Check submission format
        assert (
            "id" in df_sub.columns and "is_iceberg" in df_sub.columns
        ), "Submission columns missing"
        assert len(df_sub) > 0, "Submission file is empty"
        assert (df_sub["is_iceberg"] >= 0).all() and (
            df_sub["is_iceberg"] <= 1
        ).all(), "Predictions out of range"
    else:
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    print("Full pipeline verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
