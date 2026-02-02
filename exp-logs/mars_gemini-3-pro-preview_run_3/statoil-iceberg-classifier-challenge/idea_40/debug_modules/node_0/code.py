import os
import shutil
import sys
import numpy as np
import pandas as pd
import torch

# Import library components
from library.config import Config
from library.utils import set_seed
from library.data import get_loaders, get_test_loader
from library.model import IcebergModel
from library.train import Trainer


def run_demo():
    print("=== Starting Iceberg Classifier Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Set random seed for reproducibility
    set_seed(42)

    # Modify Config for a fast run
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8  # Smaller batch size for demo
    Config.WORKING_DIR = "./working/demo_run"

    # Re-define paths based on new WORKING_DIR
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Epochs: {Config.EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Get loaders (Debug mode implies datasets are truncated to 32 samples)
    train_loader, val_loader = get_loaders(
        batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )
    test_loader = get_test_loader(batch_size=Config.BATCH_SIZE, debug=Config.DEBUG)

    # Fetch one batch from train loader
    images, angles, targets = next(iter(train_loader))

    print(f"    Train Batch - Images Shape: {images.shape}")
    print(f"    Train Batch - Angles Shape: {angles.shape}")
    print(f"    Train Batch - Targets Shape: {targets.shape}")

    # Assertions
    expected_img_shape = (Config.BATCH_SIZE, 3, 75, 75)
    assert (
        images.shape == expected_img_shape
    ), f"Expected image shape {expected_img_shape}, got {images.shape}"
    assert angles.shape == (
        Config.BATCH_SIZE,
    ), f"Expected angles shape {(Config.BATCH_SIZE,)}, got {angles.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Expected targets shape {(Config.BATCH_SIZE,)}, got {targets.shape}"
    assert images.dtype == torch.float32, "Images should be float32"

    print("    Data Loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = IcebergModel()

    # Move to CPU for this quick check to avoid GPU overhead if not needed,
    # though Trainer handles device automatically.
    model.eval()

    with torch.no_grad():
        logits = model(images, angles)

    print(f"    Model Output (Logits) Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
    ), f"Expected logits output shape {(Config.BATCH_SIZE,)}, got {logits.shape}"

    print("    Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (Trainer.fit)...")

    # Initialize Trainer
    trainer = Trainer(debug=Config.DEBUG)

    # Run fit
    trainer.fit()

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pth")
    assert os.path.exists(
        best_model_path
    ), f"Model checkpoint not found at {best_model_path}"

    print("    Training loop completed and checkpoint saved.")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n[5] Running Inference (Trainer.predict)...")

    # Run prediction
    trainer.predict()

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Load submission and check format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission Shape: {df_sub.shape}")
    print(f"    Submission Columns: {df_sub.columns.tolist()}")

    # In debug mode, test set is truncated to 32 samples
    assert len(df_sub) == 32, f"Expected 32 rows in debug submission, got {len(df_sub)}"
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission missing required columns"
    assert df_sub["is_iceberg"].dtype == float, "is_iceberg column should be float"

    print("    Inference verification passed.")

    # -------------------------------------------------------------------------
    # 6. Cleanup
    # -------------------------------------------------------------------------
    print("\n[6] Cleaning up...")
    # Optional: Remove the demo directory to save space, or keep it for inspection.
    # We will keep it but print the location.
    print(f"    Demo artifacts stored in: {Config.WORKING_DIR}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
