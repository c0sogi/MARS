import os
import shutil
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library files
from library.config import Config
from library.data import get_dataloaders
from library.model import OCPNet
from library.train import Trainer
from library.utils import seed_everything, laplace_log_likelihood, load_checkpoint


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed & Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Override Config to run quickly and use a temp directory
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Re-run setup to create the new directories
    Config.setup()
    seed_everything(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Fetch one batch from training loader
    batch = next(iter(train_loader))
    imgs, tab, t_rel, target, pids, current_weeks = batch

    # Verify Shapes
    # Image: (Batch, 3, 224, 224) - as defined in Config.IMG_SIZE
    assert imgs.dim() == 4, f"Image tensor should be 4D, got {imgs.dim()}"
    assert (
        imgs.shape[1] == 3
    ), f"Image tensor should have 3 channels, got {imgs.shape[1]}"
    assert (
        imgs.shape[2] == Config.IMG_SIZE and imgs.shape[3] == Config.IMG_SIZE
    ), f"Image size mismatch. Expected {Config.IMG_SIZE}, got {imgs.shape[2]}x{imgs.shape[3]}"

    # Tabular: (Batch, 4) - [Base_FVC_Norm, Age_Norm, Sex, Smoke]
    assert tab.dim() == 2, f"Tabular tensor should be 2D, got {tab.dim()}"
    assert tab.shape[1] == 4, f"Tabular features should be 4, got {tab.shape[1]}"

    # Time: (Batch, 1)
    assert (
        t_rel.dim() == 2 and t_rel.shape[1] == 1
    ), f"Time tensor shape mismatch: {t_rel.shape}"

    # Target: (Batch, 1)
    assert (
        target.dim() == 2 and target.shape[1] == 1
    ), f"Target tensor shape mismatch: {target.shape}"

    print("Data Batch Shapes Verified:")
    print(f"  Images: {imgs.shape}")
    print(f"  Tabular: {tab.shape}")
    print(f"  Time: {t_rel.shape}")
    print(f"  Target: {target.shape}")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = OCPNet().to(Config.DEVICE)

    # Move batch to device
    imgs = imgs.to(Config.DEVICE)
    tab = tab.to(Config.DEVICE)
    t_rel = t_rel.to(Config.DEVICE)

    # Forward Pass
    mu, sigma = model(imgs, tab, t_rel)

    # Verify Output Shapes
    assert mu.shape == (Config.BATCH_SIZE, 1), f"Mean output shape mismatch: {mu.shape}"
    assert sigma.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Sigma output shape mismatch: {sigma.shape}"

    # Verify Sigma Positivity (Softplus ensures > 0)
    assert (sigma >= 0).all(), "Sigma (uncertainty) must be non-negative"

    print("Model Forward Pass Successful.")
    print(f"  Mu (Mean) Shape: {mu.shape}")
    print(f"  Sigma (Uncertainty) Shape: {sigma.shape}")

    # -------------------------------------------------------------------------
    # 4. Metric Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Metric Calculation...")

    # Manual Calculation Case
    # True: 2000, Pred: 2000, Sigma: 100
    # Delta = 0, Sigma_clipped = 100
    # Metric = - (sqrt(2)*0)/100 - ln(sqrt(2)*100)
    #        = 0 - ln(141.421356)
    #        = -4.95174...

    true_val = np.array([2000.0])
    pred_val = np.array([2000.0])
    sigma_val = np.array([100.0])

    score = laplace_log_likelihood(true_val, pred_val, sigma_val)
    expected_score = -np.log(np.sqrt(2) * 100)

    assert np.isclose(
        score, expected_score, atol=1e-4
    ), f"Metric calculation mismatch. Got {score}, expected {expected_score}"

    print(f"Metric Verified. Score for perfect prediction with sigma=100: {score:.4f}")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Training Loop (1 Epoch)...")

    # Initialize Trainer
    trainer = Trainer()

    # Run fit (train + validate)
    # We use the small loaders created earlier
    trainer.fit(train_loader, val_loader)

    # Check if checkpoint was created
    best_ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_ckpt_path), "Best model checkpoint was not created."

    print("Training loop completed successfully.")
    print(f"Checkpoint saved at: {best_ckpt_path}")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission Verification
    # -------------------------------------------------------------------------
    print("\n[6] Demonstrating Submission Generation...")

    # Generate submission using the test loader
    trainer.generate_submission(test_loader)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {sub_df.shape}")

    # Check Columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in sub_df.columns for col in expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check Data Integrity
    assert not sub_df.isnull().values.any(), "Submission contains NaN values."
    assert (
        sub_df["Confidence"] >= 70
    ).all(), "Confidence values should be clipped at 70."

    print("Submission file verified successfully.")
    print(sub_df.head())

    # -------------------------------------------------------------------------
    # 7. Cleanup
    # -------------------------------------------------------------------------
    print("\n[7] Cleaning up...")
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
        print(f"Removed temporary directory: {Config.WORKING_DIR}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
