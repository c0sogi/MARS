import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 1. Setup and Configuration Override
# We import Config first to modify it before other modules use it.
from library.config import Config

# Modify Config for a fast demo run
Config.NUM_EPOCHS = 1
Config.BATCH_SIZE = 16
Config.WORKING_DIR = "./working/demo_usage"
Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

# Re-run setup to create these new directories
Config.setup()

# Import library modules after config setup
from library import utils
from library import data_loader
from library import model
from library import trainer
from library import inference


def run_demo():
    print("=== Starting Library Demo ===")

    # Ensure reproducibility
    utils.set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 1. Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n[Step 1] Testing Data Processing...")

    # We force load_cached_data=False initially to test the processing logic,
    # but since we changed the cache dir, it would process anyway.
    X_train, y_train, ang_train, X_test, ang_test, ids_test = data_loader.process_data(
        load_cached_data=False
    )

    # Assertions to verify data shapes
    print(f"Train Data Shape: {X_train.shape}")
    print(f"Test Data Shape: {X_test.shape}")

    # Expected shapes: (N, 3, 75, 75)
    assert len(X_train.shape) == 4
    assert X_train.shape[1] == 3
    assert X_train.shape[2] == 75
    assert X_train.shape[3] == 75
    assert len(y_train) == len(X_train)
    assert len(ang_train) == len(X_train)

    print("Data processing verification passed.")

    # -------------------------------------------------------------------------
    # 2. Data Loaders
    # -------------------------------------------------------------------------
    print("\n[Step 2] Testing Data Loaders...")
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        fold=0, load_cached_data=True
    )

    # Fetch one batch to verify
    images, angles, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Angle Shape: {angles.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (Config.BATCH_SIZE, 3, 75, 75)
    assert angles.shape == (Config.BATCH_SIZE,)
    assert labels.shape == (Config.BATCH_SIZE,)

    print("Data loader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture
    # -------------------------------------------------------------------------
    print("\n[Step 3] Testing Model Architecture...")
    net = model.DPSCACNN().to(device)

    # Count parameters
    num_params = utils.count_parameters(net)
    print(f"Model Parameters: {num_params}")
    assert num_params > 0

    # Forward pass verification
    # Move batch to device
    dummy_images = images.to(device)
    dummy_angles = angles.to(device)

    with torch.no_grad():
        outputs = net(dummy_images, dummy_angles)

    print(f"Output Shape: {outputs.shape}")
    # Output should be (Batch_Size, 1) because NUM_CLASSES=1 (binary classification logits)
    assert outputs.shape == (Config.BATCH_SIZE, 1)

    print("Model forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Routine
    # -------------------------------------------------------------------------
    print("\n[Step 4] Testing Training Routine (Fold 0)...")

    # We run the provided run_fold function.
    # Config.NUM_EPOCHS is set to 1, so this should be quick.
    best_val_loss = trainer.run_fold(fold=0, load_cached_data=True)

    print(f"Training finished. Best Val Loss: {best_val_loss}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_fold_0.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
    print(f"Checkpoint verified at: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 5. Inference Routine
    # -------------------------------------------------------------------------
    print("\n[Step 5] Testing Inference Routine...")

    # Run inference. This will look for model_fold_{0..4}.pth.
    # We only trained fold 0. The inference script handles missing folds by skipping them
    # or we can mock the other folds by copying the checkpoint to avoid errors if the script enforces it.
    # Looking at library/inference.py, it loops folds and checks `if not os.path.exists... continue`.
    # However, it divides by `models_loaded`. So one fold is sufficient.

    inference.predict_test(load_cached_data=True)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verify submission content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission Shape: {df_sub.shape}")
    print(df_sub.head())

    # Check if number of rows matches test set size
    # We know ids_test length from step 1
    assert len(df_sub) == len(ids_test)
    assert "id" in df_sub.columns
    assert "is_iceberg" in df_sub.columns

    # Check probabilities range
    probs = df_sub["is_iceberg"].values
    assert np.all(probs >= 0.0) and np.all(
        probs <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("Inference verification passed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure clean slate for demo
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    try:
        run_demo()
    except Exception as e:
        print(f"\nDemo Failed with error: {e}")
        # Re-raise to ensure the task fails if the code is broken
        raise e
