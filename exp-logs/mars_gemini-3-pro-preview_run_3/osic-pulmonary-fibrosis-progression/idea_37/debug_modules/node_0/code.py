import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders, get_test_loader
from library.model import RCOSRNet
from library.train import Trainer
from library.inference import predict_test_set


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring Environment...")

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5  # Use only 5 patients for speed
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size for the demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead/errors in simple script

    # Set a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Setup environment (creates directories, sets seeds)
    seed_everything(Config.SEED)
    Config.setup()

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Loading...")

    # Get dataloaders in debug mode
    train_loader, val_loader = get_dataloaders(
        debug=True, batch_size=Config.BATCH_SIZE, num_workers=0
    )

    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches: {len(val_loader)}")

    # Fetch one batch to verify structure and shapes
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = {
        "image",
        "clinical",
        "patient_id",
        "weeks",
        "target",
        "raw_fvc",
        "patient_week",
    }
    assert expected_keys.issubset(
        batch.keys()
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Verify Shapes
    # Image: (B, 3, 260, 260)
    imgs = batch["image"]
    assert imgs.dim() == 4, f"Image tensor should be 4D, got {imgs.dim()}"
    assert imgs.shape[1] == 3, f"Image should have 3 channels, got {imgs.shape[1]}"
    assert (
        imgs.shape[2] == Config.IMG_SIZE and imgs.shape[3] == Config.IMG_SIZE
    ), f"Image size mismatch. Expected {Config.IMG_SIZE}x{Config.IMG_SIZE}, got {imgs.shape[2]}x{imgs.shape[3]}"

    # Clinical: (B, 5)
    clinical = batch["clinical"]
    assert clinical.dim() == 2, f"Clinical tensor should be 2D, got {clinical.dim()}"
    assert (
        clinical.shape[1] == Config.CLINICAL_INPUT_DIM
    ), f"Clinical input dim mismatch. Expected {Config.CLINICAL_INPUT_DIM}, got {clinical.shape[1]}"

    # Target: (B)
    target = batch["target"]
    assert target.dim() == 1, f"Target tensor should be 1D, got {target.dim()}"

    print("Data Loader verification successful.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 3] Initializing Model and Running Forward Pass...")

    model = RCOSRNet().to(Config.DEVICE)

    # Move batch to device
    imgs = imgs.to(Config.DEVICE)
    clinical = clinical.to(Config.DEVICE)

    # Forward pass
    mu, sigma = model(imgs, clinical)

    # Verify Output Shapes: (B, 1)
    assert mu.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Output Mean shape mismatch. Got {mu.shape}"
    assert sigma.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Output Sigma shape mismatch. Got {sigma.shape}"

    # Verify Sigma Positivity (Softplus ensures > 0)
    assert (sigma > 0).all(), "Sigma predictions must be positive."

    print("Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Training Loop (1 Epoch)...")

    trainer = Trainer(model, train_loader, val_loader, Config.DEVICE)

    # Fit model
    trainer.fit(epochs=Config.EPOCHS)

    # Verify Checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Model checkpoint was not saved."
    print(f"Checkpoint verified at: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Generating Submission...")

    # Use the inference module to generate predictions
    # This function loads the best model from checkpoint automatically if not passed,
    # but we can pass the trained model directly to save time reloading.
    submission_df = predict_test_set(model=model, device=Config.DEVICE)

    # Verify Submission File
    sub_file_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_file_path), "Submission file was not created."

    # Verify Content
    assert "Patient_Week" in submission_df.columns
    assert "FVC" in submission_df.columns
    assert "Confidence" in submission_df.columns

    # Check row count matches sample submission
    sample_sub = pd.read_csv(os.path.join(Config.INPUT_DIR, "sample_submission.csv"))
    assert len(submission_df) == len(
        sample_sub
    ), f"Submission row count mismatch. Expected {len(sample_sub)}, got {len(submission_df)}"

    # Check for NaNs
    assert not submission_df.isnull().values.any(), "Submission contains NaN values."

    print(f"Submission generated successfully with {len(submission_df)} rows.")
    print(submission_df.head())

    # -------------------------------------------------------------------------
    # 6. Metric Calculation Check
    # -------------------------------------------------------------------------
    print("\n[Step 6] Verifying Metric Calculation...")

    # Create dummy data
    y_true = np.array([2000, 2500, 3000])
    y_pred = np.array([2100, 2400, 2000])  # Errors: 100, 100, 1000
    y_std = np.array([50, 100, 200])  # 50 will be clipped to 70

    score = calculate_metric(y_true, y_pred, y_std)

    # Manual Calculation for first element:
    # Delta = |2000 - 2100| = 100
    # Sigma_clipped = max(50, 70) = 70
    # Metric = - (sqrt(2) * 100 / 70) - ln(sqrt(2) * 70)
    #        = - (1.4142 * 1.428) - ln(98.99)
    #        = - 2.02 - 4.59 = -6.61 approx

    assert isinstance(score, float), "Metric should return a float."
    assert score < 0, "Metric should be negative."

    print(f"Calculated Metric on dummy data: {score:.4f}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
