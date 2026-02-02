import os
import shutil
import torch
import numpy as np
import pandas as pd
import time

# Import from the provided library
from library.configuration import Config
from library.utilities import set_seed, get_or_create_cached_array
from library.data_loader import get_dataloaders, get_data_arrays
from library.architecture import IcebergResNet
from library.optimization import ConsistencyLoss
from library.experiment_manager import ExperimentManager


def run_demo():
    print("Starting Iceberg Classifier Library Demo...")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config paths to use a demo directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Override Hyperparameters for speed
    Config.MAX_EPOCHS_PHASE_1 = 1  # Run only 1 epoch for calibration
    Config.EARLY_STOPPING_PATIENCE = 1
    Config.NUM_ENSEMBLE_MODELS = 1  # Train only 1 model for production
    Config.SWA_EPOCHS = 1  # Run only 1 SWA epoch
    Config.BATCH_SIZE = 16  # Smaller batch size

    # Set Seed
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Load raw arrays first to check shapes
    train_imgs, train_angles, train_labels, test_imgs, test_angles, test_ids = (
        get_data_arrays(load_cached_data=False)
    )

    print(f"Train Images Shape: {train_imgs.shape}")
    print(f"Train Angles Shape: {train_angles.shape}")
    print(f"Train Labels Shape: {train_labels.shape}")

    assert train_imgs.shape[1:] == (
        224,
        224,
        3,
    ), "Incorrect image dimensions after preprocessing"
    assert len(train_imgs) == len(train_labels), "Mismatch between images and labels"

    # Get DataLoaders
    loaders = get_dataloaders(load_cached_data=True)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # Verify Train Loader (Dual-View)
    # Train loader returns: ((img1, img2), angle, label)
    batch = next(iter(train_loader))
    (img1, img2), angles, labels = batch

    print(f"Batch Size: {img1.size(0)}")
    assert img1.size() == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect batch image shape (View 1)"
    assert img2.size() == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), "Incorrect batch image shape (View 2)"
    assert angles.size(0) == Config.BATCH_SIZE, "Incorrect angle batch size"
    assert labels.size(0) == Config.BATCH_SIZE, "Incorrect label batch size"

    print("Data Loader verification successful.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = IcebergResNet().to(Config.DEVICE)

    # Move batch to device
    img1_dev = img1.to(Config.DEVICE)
    angles_dev = angles.to(Config.DEVICE)

    # Forward Pass
    logits = model(img1_dev, angles_dev)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"

    print("Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Consistency Loss...")

    criterion = ConsistencyLoss()

    # Create dummy logits for two views
    logits1 = torch.randn(
        Config.BATCH_SIZE, 1, device=Config.DEVICE, requires_grad=True
    )
    logits2 = torch.randn(
        Config.BATCH_SIZE, 1, device=Config.DEVICE, requires_grad=True
    )
    targets = torch.randint(0, 2, (Config.BATCH_SIZE,), device=Config.DEVICE).float()

    loss = criterion(logits1, logits2, targets)

    print(f"Loss Value: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    # Verify backward pass capability
    loss.backward()
    assert logits1.grad is not None, "Gradients not computed for logits1"

    print("Loss function verification successful.")

    # -------------------------------------------------------------------------
    # 5. Full Pipeline Execution (Experiment Manager)
    # -------------------------------------------------------------------------
    print(
        "\n[5] Running Experiment Manager (Calibration -> Production -> Inference)..."
    )

    manager = ExperimentManager()

    start_time = time.time()
    manager.execute()
    elapsed = time.time() - start_time

    print(f"Experiment execution completed in {elapsed:.2f} seconds.")

    # -------------------------------------------------------------------------
    # 6. Submission Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Submission File...")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission loaded. Shape: {df_sub.shape}")
        print("First 5 rows:")
        print(df_sub.head())

        # Basic validation
        assert (
            "id" in df_sub.columns and "is_iceberg" in df_sub.columns
        ), "Missing columns in submission"
        assert df_sub.shape[0] > 0, "Submission file is empty"
        assert (
            df_sub["is_iceberg"].min() >= 0 and df_sub["is_iceberg"].max() <= 1
        ), "Probabilities out of bounds"

        print("Submission file verification successful.")
    else:
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
