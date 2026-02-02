import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, calculate_global_stats
from library.data_loader import process_data, IcebergDataset, get_transforms
from library.model import RIWBN
from library.trainer import run_training


def demo_usage():
    print("=== Starting Demonstration of Iceberg Classification Library ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Fast Demonstration
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid execution...")

    # Override Config parameters to ensure the script runs quickly
    Config.DEBUG = True
    Config.MAX_SAMPLES = 50  # Use only 50 samples for this demo
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch per fold
    Config.NUM_FOLDS = 2  # Use only 2 folds
    Config.BATCH_SIZE = 8  # Small batch size
    Config.PATIENCE = 1  # Aggressive early stopping

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Max Samples: {Config.MAX_SAMPLES}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading and Processing Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading and Processing...")

    # A. Process Data (Load JSONs, align with metadata, cache results)
    # This function handles the heavy lifting of parsing the JSONs
    data_artifacts = process_data(load_cached_data=False)  # Force re-process for demo

    # Verify keys exist
    expected_keys = [
        "X_train",
        "inc_train",
        "y_train",
        "X_val",
        "inc_val",
        "y_val",
        "X_test",
        "inc_test",
    ]
    for key in expected_keys:
        assert key in data_artifacts, f"Missing key in processed data: {key}"

    # Verify Shapes
    # X shape should be (N, 3, 75, 75)
    X_train = data_artifacts["X_train"]
    y_train = data_artifacts["y_train"]
    print(f"Train Data Shape: {X_train.shape}")
    print(f"Train Label Shape: {y_train.shape}")

    assert len(X_train) == len(y_train), "Mismatch between X and y lengths"
    assert X_train.shape[1:] == (
        3,
        75,
        75,
    ), f"Unexpected image dimensions: {X_train.shape}"

    # B. Calculate Global Stats
    # Used for normalization
    stats = calculate_global_stats(load_cached_data=False, debug=Config.DEBUG)
    print(f"Global Stats: {stats}")
    assert "b1_min" in stats and "b1_max" in stats, "Stats dictionary missing keys"

    # Add b3 stats manually as done in trainer.py (since b3 is derived)
    # In a real scenario, this logic is inside run_training, but we replicate for unit testing
    b3_data = X_train[:, 2, :, :]
    stats["b3_min"] = float(b3_data.min())
    stats["b3_max"] = float(b3_data.max())

    # C. Dataset and DataLoader Verification
    print("Verifying Dataset and DataLoader...")
    train_ds = IcebergDataset(
        X=X_train,
        inc_angles=data_artifacts["inc_train"],
        labels=y_train,
        transform=get_transforms("train"),
        global_stats=stats,
    )

    # Fetch a single item
    img, inc, label = train_ds[0]
    print(
        f"Single Item Shapes -> Img: {img.shape}, Inc: {inc.shape}, Label: {label.shape}"
    )

    assert img.shape == (3, 75, 75), "Dataset image shape incorrect"
    assert isinstance(img, torch.Tensor), "Dataset should return tensors"
    assert not torch.isnan(img).any(), "Image tensor contains NaNs"

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture (RIWBN)...")

    model = RIWBN().to(Config.DEVICE)
    model.train()

    # Create dummy batch
    batch_size = 4
    dummy_img = torch.randn(batch_size, 3, 75, 75).to(Config.DEVICE)
    dummy_inc = torch.randn(batch_size).to(Config.DEVICE)

    # Forward Pass
    output = model(dummy_img, dummy_inc)
    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output (B, 1), got {output.shape}"

    # Backward Pass Check (Gradient Flow)
    target = torch.randint(0, 2, (batch_size, 1)).float().to(Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(output, target)
    loss.backward()

    # Check if gradients exist for a specific layer
    conv_grad = model.stage1_conv.weight.grad
    assert conv_grad is not None, "Gradients not computed for first layer"
    print("Forward and Backward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Full Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[4] Running Full Training Pipeline (Simulated)...")

    # Clean up previous runs in working directory if needed
    # (Config.setup() handles creation, but we want a fresh start for the demo log)
    if os.path.exists(Config.SUBMISSION_PATH):
        os.remove(Config.SUBMISSION_PATH)

    # Execute the training routine provided in library.trainer
    # This will:
    # 1. Load data (using the cached/processed data we verified)
    # 2. Run Stratified K-Fold Cross Validation
    # 3. Train models
    # 4. Generate predictions on Test set
    # 5. Save submission.csv
    run_training()

    # -------------------------------------------------------------------------
    # 5. Output Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Submission Output...")

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file found at {Config.SUBMISSION_PATH}")
        print(f"Submission Shape: {df_sub.shape}")
        print("First 5 rows:")
        print(df_sub.head())

        # Verify submission format
        assert list(df_sub.columns) == [
            "id",
            "is_iceberg",
        ], "Incorrect columns in submission"
        assert df_sub["is_iceberg"].min() >= 0.0, "Probabilities should be >= 0"
        assert df_sub["is_iceberg"].max() <= 1.0, "Probabilities should be <= 1"

        # Verify row count matches test set size (limited by MAX_SAMPLES/DEBUG logic)
        # Note: In debug mode, test set is also truncated by process_data logic
        # process_data truncates raw IDs.
        print("Submission verification passed.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    demo_usage()
