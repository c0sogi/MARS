import os
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, loss_fn
from library.data import get_dataloaders
from library.model import PGARNet
from library.train import run_training
from library.predict import generate_submission


def run_demo():
    print("=== Starting OSIC Pulmonary Fibrosis Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Override paths to keep demo artifacts separate
    Config.OUTPUT_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.OUTPUT_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.OUTPUT_DIR, "checkpoints")
    Config.SUBMISSION_DIR = Config.OUTPUT_DIR  # Save submission in root of demo dir

    # Override hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.MAX_TRAIN_SAMPLES = 12  # Small subset for training
    Config.MAX_VAL_SAMPLES = 8  # Small subset for validation

    # Re-run setup to create directories and set seeds with new config
    Config.setup()

    # ---------------------------------------------------------
    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n[Step 2] Verifying Data Pipeline...")

    # Load data loaders (this will trigger caching for the small subset)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))
    inputs, targets = batch

    # Unpack inputs
    axial = inputs["axial"]
    coronal = inputs["coronal"]
    tabular = inputs["tabular"]
    dt = inputs["dt"]
    base_fvc = inputs["base_fvc"]

    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Axial Image Shape: {axial.shape}")
    print(f"  Coronal Image Shape: {coronal.shape}")
    print(f"  Tabular Vector Shape: {tabular.shape}")

    # Assertions
    assert axial.shape == (Config.BATCH_SIZE, 3, 224, 224), "Incorrect Axial shape"
    assert coronal.shape == (Config.BATCH_SIZE, 3, 224, 224), "Incorrect Coronal shape"
    assert tabular.shape == (
        Config.BATCH_SIZE,
        7,
    ), "Incorrect Tabular shape (Age, Percent, Sex*2, Smoke*3)"
    assert dt.shape == (Config.BATCH_SIZE,), "Incorrect dt shape"
    assert targets.shape == (Config.BATCH_SIZE,), "Incorrect target shape"

    print("  -> Data Pipeline verified successfully.")

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying PGARNet Architecture...")

    device = Config.DEVICE
    model = PGARNet().to(device)

    # Move batch to device
    axial = axial.to(device)
    coronal = coronal.to(device)
    tabular = tabular.to(device)
    dt = dt.to(device)
    base_fvc = base_fvc.to(device)
    targets = targets.to(device)

    # Forward pass
    fvc_pred, sigma_pred = model(axial, coronal, tabular, dt, base_fvc)

    print(f"  Prediction Shape (FVC): {fvc_pred.shape}")
    print(f"  Prediction Shape (Sigma): {sigma_pred.shape}")

    # Assertions
    assert fvc_pred.shape == (Config.BATCH_SIZE,), "Output FVC shape mismatch"
    assert sigma_pred.shape == (Config.BATCH_SIZE,), "Output Sigma shape mismatch"
    assert torch.all(sigma_pred > 0), "Sigma predictions must be positive (Softplus)"

    print("  -> Model Forward Pass verified successfully.")

    # ---------------------------------------------------------
    # 4. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[Step 4] Verifying Loss Function...")

    loss = loss_fn(targets, fvc_pred, sigma_pred)

    print(f"  Calculated Loss: {loss.item():.4f}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.dim() == 0, "Loss must be a scalar"

    print("  -> Loss Function verified successfully.")

    # ---------------------------------------------------------
    # 5. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[Step 5] Executing Training Loop (Demo Mode)...")

    # Run training with reduced epochs
    best_score = run_training(
        epochs=Config.EPOCHS,
        lr=1e-3,  # Slightly higher LR to see changes if we were monitoring
        load_cached_data=True,
    )

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not created at {checkpoint_path}")

    print(f"  -> Training complete. Best Score: {best_score}")
    print(f"  -> Checkpoint verified at: {checkpoint_path}")

    # ---------------------------------------------------------
    # 6. Inference and Submission
    # ---------------------------------------------------------
    print("\n[Step 6] Generating Submission...")

    # Generate submission using the trained model
    # Note: We use the same batch size as config
    submission_df = generate_submission(load_cached_data=True)

    # Verify Submission
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}"

    # Check if file exists
    sub_file_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(sub_file_path):
        raise FileNotFoundError(f"Submission file not found at {sub_file_path}")

    print(f"  -> Submission generated with {len(submission_df)} rows.")
    print("  -> Inference pipeline verified successfully.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
