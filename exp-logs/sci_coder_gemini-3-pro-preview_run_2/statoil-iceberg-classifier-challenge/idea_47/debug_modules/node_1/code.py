import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Ensure the current directory is in the python path for library imports
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.data_handling import load_data, get_dataloaders, IcebergDataset
from library.model import PCWBN
from library.training import train_k_fold, predict_and_submit


def run_demo():
    print("Starting PC-WBN Library Demo...")

    # ==========================================
    # 1. SETUP & CONFIGURATION OVERRIDE
    # ==========================================
    # We modify the Config class attributes at runtime to create a lightweight demo environment.
    print("\n[1] Configuring environment...")

    # Set a specific working directory for this demo
    DEMO_WORK_DIR = "./working/demo_run"
    if os.path.exists(DEMO_WORK_DIR):
        shutil.rmtree(DEMO_WORK_DIR)
    os.makedirs(DEMO_WORK_DIR, exist_ok=True)

    # Override Config constants
    Config.WORK_DIR = DEMO_WORK_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_WORK_DIR, "submission")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.CACHE_FILE = os.path.join(DEMO_WORK_DIR, "cache", "processed_data.npz")

    # Reduce complexity for speed
    Config.NUM_FOLDS = 2
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo

    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.CACHE_FILE), exist_ok=True)

    seed_everything(Config.SEED)
    print(f"    Working Directory: {Config.WORK_DIR}")
    print(f"    Epochs: {Config.NUM_EPOCHS}, Folds: {Config.NUM_FOLDS}")

    # ==========================================
    # 2. DATA HANDLING VERIFICATION
    # ==========================================
    print("\n[2] Verifying Data Loading...")

    # Load data (this will trigger caching)
    data = load_data(load_cached_data=False)

    X_train = data["X_train"]
    inc_train = data["inc_train"]
    y_train = data["y_train"]
    stats = data["stats"]

    # Assertions
    print(f"    Train Data Shape: {X_train.shape}")
    assert len(X_train.shape) == 4, "X_train should be 4D (N, C, H, W)"
    assert X_train.shape[1] == 3, "Should have 3 channels (Band1, Band2, Mean)"
    assert (
        X_train.shape[2] == 75 and X_train.shape[3] == 75
    ), "Image dimensions should be 75x75"
    assert len(inc_train) == len(X_train), "Incidence angle count mismatch"
    assert len(y_train) == len(X_train), "Label count mismatch"

    print("    Global Stats Check:")
    for k, v in stats.items():
        print(f"      {k}: {v:.4f}")
        assert isinstance(v, float), f"Stat {k} should be a float"

    # ==========================================
    # 3. DATALOADER VERIFICATION
    # ==========================================
    print("\n[3] Verifying DataLoaders...")

    # Get loaders with debug size
    DEBUG_SIZE = 16
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug_size=DEBUG_SIZE,
    )

    # Fetch one batch
    images, inc_angles, labels = next(iter(train_loader))

    print(
        f"    Batch Shapes -> Images: {images.shape}, Inc: {inc_angles.shape}, Labels: {labels.shape}"
    )

    # Assertions
    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75), "Incorrect image batch shape"
    assert inc_angles.shape == (
        Config.BATCH_SIZE,
    ), "Incorrect incidence angle batch shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label batch shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert not torch.isnan(images).any(), "Images contain NaNs"

    # ==========================================
    # 4. MODEL ARCHITECTURE VERIFICATION
    # ==========================================
    print("\n[4] Verifying PC-WBN Model...")

    device = Config.DEVICE
    model = PCWBN().to(device)

    # Move batch to device
    images = images.to(device)
    inc_angles = inc_angles.to(device)

    # Forward pass
    logits = model(images, inc_angles)

    print(f"    Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (Config.BATCH_SIZE, 1), "Output should be (Batch_Size, 1)"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    # Clean up GPU memory
    del model, images, inc_angles, logits
    torch.cuda.empty_cache()

    # ==========================================
    # 5. TRAINING PIPELINE VERIFICATION
    # ==========================================
    print("\n[5] Running Training Pipeline (K-Fold)...")

    # Run training with debug size
    # This tests: Data loading, Model init, Forward/Backward pass, Saving checkpoints
    trained_stats = train_k_fold(debug_size=DEBUG_SIZE, epochs=Config.NUM_EPOCHS)

    # Verify artifacts
    expected_model_path = os.path.join(Config.WORK_DIR, "model_fold_0.pth")
    print(f"    Checking for artifact: {expected_model_path}")
    assert os.path.exists(expected_model_path), "Model checkpoint for Fold 0 not found"

    # ==========================================
    # 6. PREDICTION PIPELINE VERIFICATION
    # ==========================================
    print("\n[6] Running Prediction Pipeline...")

    # Run prediction
    predict_and_submit(trained_stats, debug_size=DEBUG_SIZE)

    # Verify submission
    print(f"    Checking for submission: {Config.SUBMISSION_FILE}")
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission Shape: {df_sub.shape}")
    print(f"    Columns: {df_sub.columns.tolist()}")

    # Assertions
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission columns missing"
    assert (
        len(df_sub) == DEBUG_SIZE
    ), f"Submission should have {DEBUG_SIZE} rows (debug mode)"
    assert (
        df_sub["is_iceberg"].min() >= 0.0 and df_sub["is_iceberg"].max() <= 1.0
    ), "Probabilities out of range"

    print("\nSUCCESS: All library components verified and executed correctly.")


if __name__ == "__main__":
    run_demo()
