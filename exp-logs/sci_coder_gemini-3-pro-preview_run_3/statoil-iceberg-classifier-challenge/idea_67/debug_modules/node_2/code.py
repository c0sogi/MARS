import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import logging

# Import library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.model as model_lib
import library.train as train_lib


def run_demo():
    print("=== Starting Idea 67 Demo Execution ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    print("\n[1] Patching Configuration for Demo...")

    # Redirect working directories to avoid overwriting real training artifacts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    config.WORKING_DIR = DEMO_DIR
    config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Reduce compute requirements
    config.NUM_EPOCHS = 1  # Run only 1 epoch per fold
    config.NUM_FOLDS = 2  # Run only 2 folds instead of 5
    config.BATCH_SIZE = 4  # Small batch size
    config.DEBUG = True  # Enable debug mode to slice dataset
    config.MAX_DEBUG_SAMPLES = 20  # Use only 20 samples for training/val

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Debug Mode: {config.DEBUG}")
    print(f"Max Samples: {config.MAX_DEBUG_SAMPLES}")

    # Setup Logging
    logger = utils.setup_logging("demo_execution.log")
    utils.set_seed(config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Load data (this will process raw json since cache dir is new/empty)
    # Note: load_data processes all data, but get_fold_loaders slices it if DEBUG is True
    X_train, angles_train, y_train, ids_train, X_test, angles_test, ids_test = (
        data_loader.load_data(load_cached_data=True)
    )

    # Basic Data Assertions
    assert len(X_train) == len(y_train) == len(ids_train)
    assert X_train.shape[1:] == (
        3,
        75,
        75,
    ), f"Expected (3, 75, 75), got {X_train.shape[1:]}"
    assert not np.isnan(X_train).any(), "X_train contains NaNs"

    print(f"Loaded {len(X_train)} training samples.")
    print(f"Loaded {len(X_test)} test samples.")

    # Verify Fold Loader
    train_loader, val_loader = data_loader.get_fold_loaders(
        fold_idx=0, load_cached_data=True
    )

    # Check Batch Structure
    batch = next(iter(train_loader))
    images = batch["image"]
    angles = batch["angle"]
    labels = batch["label"]
    ids = batch["id"]

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Angle Shape: {angles.shape}")

    assert images.shape == (config.BATCH_SIZE, 3, 75, 75)
    assert angles.shape == (config.BATCH_SIZE,)
    assert labels.shape == (config.BATCH_SIZE,)
    assert len(ids) == config.BATCH_SIZE

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_lib.LeakyAttentiveIsomorphicCNN().to(device)

    # Create dummy input based on batch shape
    dummy_img = torch.randn(config.BATCH_SIZE, 3, 75, 75).to(device)
    dummy_ang = torch.randn(config.BATCH_SIZE).to(device)

    # Forward Pass
    logits = model(dummy_img, dummy_ang)

    print(f"Model Output Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (config.BATCH_SIZE, 1), "Model output shape mismatch"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    # -------------------------------------------------------------------------
    # 4. Training Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (Reduced Folds/Epochs)...")

    # Run training for the configured number of folds (2)
    # This calls run_fold_training internally
    losses = train_lib.train_all_folds(load_cached_data=True)

    print(f"Training completed. Losses: {losses}")
    assert len(losses) == config.NUM_FOLDS

    # Verify Checkpoints
    for i in range(config.NUM_FOLDS):
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"model_best_fold_{i}.pth")
        assert os.path.exists(
            ckpt_path
        ), f"Checkpoint for fold {i} missing at {ckpt_path}"
        print(f"Verified checkpoint: {ckpt_path}")

    # -------------------------------------------------------------------------
    # 5. Submission Generation Verification
    # -------------------------------------------------------------------------
    print("\n[5] Generating Submission...")

    train_lib.generate_submission(load_cached_data=True)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not created"

    # Verify Submission Content
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    assert "id" in df_sub.columns
    assert "is_iceberg" in df_sub.columns
    assert len(df_sub) > 0

    # Check probability range
    probs = df_sub["is_iceberg"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
