import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data_loader import process_data, get_fold_loaders, get_test_loader
from library.model import SQWBN
from library.train_eval import train_fold, predict_ensemble


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Configuring environment for fast demonstration...")
    seed_everything(42)

    # Override Config for a quick run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_FOLDS = 2  # Simulate 2 folds (we will only train fold 0)
    Config.PATIENCE = 2  # Short patience

    # Redirect outputs to a demo directory
    Config.ARTIFACT_DIR = "./working/demo_artifacts"
    Config.SUBMISSION_PATH = os.path.join(Config.ARTIFACT_DIR, "demo_submission.csv")
    Config.CACHE_PATH = os.path.join(
        Config.ARTIFACT_DIR, "cache", "processed_data_debug.npz"
    )

    # Ensure directory exists
    os.makedirs(Config.ARTIFACT_DIR, exist_ok=True)
    print(f"Artifacts will be saved to: {Config.ARTIFACT_DIR}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Processing and Loading...")

    # Test process_data (force reprocessing to verify logic)
    X_train, y_train, inc_train, X_test, inc_test, test_ids = process_data(
        load_cached_data=False
    )

    print(f"Processed Train Shape: {X_train.shape}")
    print(f"Processed Test Shape: {X_test.shape}")

    # Assertions for data processing
    assert (
        len(X_train) == len(y_train) == len(inc_train)
    ), "Train data dimension mismatch"
    assert X_train.shape[1:] == (
        3,
        75,
        75,
    ), f"Unexpected image shape: {X_train.shape[1:]}"
    assert not np.isnan(X_train).any(), "NaNs found in training data"

    # Test DataLoader
    train_loader, val_loader = get_fold_loaders(fold_idx=0, load_cached_data=True)
    images, angles, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Angle Shape: {angles.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75), "Incorrect batch image shape"
    assert angles.shape == (Config.BATCH_SIZE,), "Incorrect batch angle shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect batch label shape"

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture (SQWBN)...")
    device = torch.device(Config.DEVICE)
    model = SQWBN().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    outputs = model(images, angles)

    print(f"Model Output Shape: {outputs.shape}")
    print(f"Sample Prediction: {outputs[0].item():.4f}")

    assert outputs.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert (
        0.0 <= outputs.min().item() and outputs.max().item() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    # 4. Training Loop Demonstration
    print("\n[4] Demonstrating Training Loop (Fold 0)...")
    # Train fold 0
    best_score = train_fold(fold_idx=0)

    # Verify model artifact creation
    model_path = os.path.join(Config.ARTIFACT_DIR, "model_fold_0.pth")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    print(f"Training successful. Best Validation Loss: {best_score:.4f}")

    # 5. Inference and Submission Demonstration
    print("\n[5] Demonstrating Ensemble Prediction...")
    # This will look for model_fold_0.pth (and skip others since they don't exist)
    predict_ensemble()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Loaded. Shape: {df_sub.shape}")
    print(df_sub.head(3))

    # Assert submission format
    assert list(df_sub.columns) == ["id", "is_iceberg"], "Incorrect submission columns"
    assert len(df_sub) > 0, "Submission file is empty"
    # In debug mode, test set is not subsetted by Config.DEBUG_SUBSET_SIZE in process_data
    # (only train/val are subsetted in get_fold_loaders), but process_data returns full arrays.
    # However, get_test_loader calls process_data which returns full test set.
    # Let's verify we have rows equal to the test set size.
    assert len(df_sub) == len(
        test_ids
    ), f"Submission row count {len(df_sub)} != Test ID count {len(test_ids)}"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
