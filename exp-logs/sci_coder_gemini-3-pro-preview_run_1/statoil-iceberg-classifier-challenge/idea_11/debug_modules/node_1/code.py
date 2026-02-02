import os
import torch
import pandas as pd
import numpy as np
import shutil
from library import config, utils, data, model, engine, calibration, production


def run_demo():
    print("============================================================")
    print("DEMO: Iceberg Classifier Pipeline Execution")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[Step 1] Setting up Configuration for Demo...")

    # Override config for speed and isolation
    config.WORKING_DIR = "./working/demo_run"
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "submission.csv")

    # Reduce computational load for demonstration
    config.CALIBRATION_EPOCHS = 2  # Run only 2 epochs per fold
    config.NUM_FOLDS = 2  # Run only 2 folds
    config.NUM_ENSEMBLE_MODELS = 2  # Train only 2 models for the ensemble
    config.BATCH_SIZE = 16  # Smaller batch size

    # Ensure directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set seeds
    utils.seed_everything(config.RANDOM_SEED)
    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Device: {config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Loading...")

    # Force processing of raw data (load_cached_data=False) to test parsing logic
    # This will save .npy files to the new config.WORKING_DIR
    train_loader, val_loader, test_loader = data.get_dataloaders(
        load_cached_data=False, full_train=False
    )

    # Fetch a single batch from the training loader
    images, angles, labels, ids = next(iter(train_loader))

    # Assertions to verify data shapes and types
    print(
        f"Batch Shapes -> Images: {images.shape}, Angles: {angles.shape}, Labels: {labels.shape}"
    )

    # Expected: (Batch, 3, 224, 224) - 3 channels because of composite band creation and upsampling
    assert images.shape == (
        config.BATCH_SIZE,
        3,
        config.IMG_SIZE,
        config.IMG_SIZE,
    ), f"Image shape mismatch. Expected ({config.BATCH_SIZE}, 3, {config.IMG_SIZE}, {config.IMG_SIZE}), got {images.shape}"

    assert angles.shape == (
        config.BATCH_SIZE,
    ), f"Angle shape mismatch. Expected ({config.BATCH_SIZE},), got {angles.shape}"

    assert labels.shape == (
        config.BATCH_SIZE,
    ), f"Label shape mismatch. Expected ({config.BATCH_SIZE},), got {labels.shape}"

    # Verify normalization (approximate range check)
    # Since we normalize and then augment, values might drift slightly, but should be roughly reasonable
    assert (
        images.min() >= -5.0 and images.max() <= 5.0
    ), "Image values seem out of expected normalized range."

    print("Data Loading verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")

    # Instantiate model
    net = model.IcebergResNet18()
    net.to(config.DEVICE)

    # Move batch to device
    images = images.to(config.DEVICE)
    angles = angles.to(config.DEVICE)

    # Forward pass
    outputs = net(images, angles)

    print(f"Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected ({config.BATCH_SIZE}, 1), got {outputs.shape}"

    # Check if outputs are finite (no NaNs)
    assert torch.isfinite(outputs).all(), "Model produced NaN or Inf values."

    print("Model Architecture verified successfully.")

    # Clean up memory
    del net, images, angles, outputs
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Phase 1: Calibration (Simulation)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Phase 1: Calibration...")

    # We use the library function but with our overridden config constants.
    # We use load_cached_data=True now because Step 2 already generated the cache in WORKING_DIR.
    best_epoch, milestones = calibration.run_calibration(
        load_cached_data=True,
        epochs=config.CALIBRATION_EPOCHS,
        n_folds=config.NUM_FOLDS,
    )

    print(f"Calibration returned -> Best Epoch: {best_epoch}, Milestones: {milestones}")

    # Assertions
    assert (
        isinstance(best_epoch, int) and best_epoch > 0
    ), "Best epoch must be a positive integer."
    assert isinstance(milestones, list), "Milestones must be a list."

    # Since we run for 2 epochs, best_epoch must be 1 or 2
    assert (
        best_epoch <= config.CALIBRATION_EPOCHS
    ), "Best epoch cannot exceed total calibration epochs."

    print("Phase 1 executed successfully.")

    # -------------------------------------------------------------------------
    # 5. Phase 2: Production (Simulation)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Phase 2: Production...")

    # Run production training and inference
    production.train_production_ensemble(
        best_epoch=best_epoch, lr_milestones=milestones, load_cached_data=True
    )

    # Verify Submission File
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), f"Submission file not found at {config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission File loaded. Shape: {df_sub.shape}")
    print(f"First few rows:\n{df_sub.head()}")

    # Assertions on Submission
    # Test set size is 321 based on metadata info provided in prompt
    EXPECTED_TEST_SIZE = 321
    assert (
        len(df_sub) == EXPECTED_TEST_SIZE
    ), f"Submission row count mismatch. Expected {EXPECTED_TEST_SIZE}, got {len(df_sub)}"

    assert list(df_sub.columns) == [
        "id",
        "is_iceberg",
    ], f"Submission columns mismatch. Expected ['id', 'is_iceberg'], got {list(df_sub.columns)}"

    # Check probability range
    probs = df_sub["is_iceberg"].values
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]."

    # Check for NaNs
    assert not df_sub.isnull().any().any(), "Submission contains NaN values."

    print("Phase 2 executed successfully.")

    print("\n============================================================")
    print("DEMO COMPLETED SUCCESSFULLY")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
