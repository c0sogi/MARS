import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import LungDataset
from library.model import IASDANet
from library.engine import train_model, predict

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Pulmonary Fibrosis Inference Demo ===")

    # 1. Setup and Configuration
    # Initialize directories and seeds
    Config.initialize()

    # Enable Debug Mode for speed
    # This sets EPOCHS=2 and limits dataset length to 32 samples
    print("Setting Debug Mode to True for fast execution...")
    Config.set_debug_mode(True)

    # Ensure reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("\n[Step 1] Initializing Datasets...")

    # Create datasets
    train_ds = LungDataset(mode="train")
    val_ds = LungDataset(mode="val")
    test_ds = LungDataset(mode="test")

    print(f"Train samples (Debug): {len(train_ds)}")
    print(f"Val samples (Debug): {len(val_ds)}")
    print(f"Test samples (Debug): {len(test_ds)}")

    # Create DataLoaders
    # We use num_workers=0 to minimize overhead for this small demo
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True,  # Drop last to ensure batch norm stability if batch is small
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 3. Verification: Data & Model Logic
    print("\n[Step 2] Verifying Data and Model Architecture...")

    # Fetch a single batch
    batch = next(iter(train_loader))

    # Move to device
    axial = batch["axial_img"].to(device)
    coronal = batch["coronal_img"].to(device)
    tabular = batch["tabular"].to(device)
    time_delta = batch["time_delta"].to(device)
    baseline_fvc = batch["baseline_fvc"].to(device)

    # Verify Input Shapes
    # Image: [B, 3, 224, 224]
    assert axial.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Incorrect Axial Image shape: {axial.shape}"
    assert coronal.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Incorrect Coronal Image shape: {coronal.shape}"
    # Tabular: [B, 6] (Age, Sex, Smoke*3, Percent)
    assert tabular.shape == (
        Config.BATCH_SIZE,
        6,
    ), f"Incorrect Tabular shape: {tabular.shape}"

    # Initialize Model
    model = IASDANet().to(device)

    # Run Forward Pass
    with torch.no_grad():
        fvc_pred, sigma_pred = model(
            axial_img=axial,
            coronal_img=coronal,
            tabular=tabular,
            time_delta=time_delta,
            baseline_fvc=baseline_fvc,
        )

    # Verify Output Shapes
    # Output should be [B, 1]
    assert fvc_pred.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect FVC Pred shape: {fvc_pred.shape}"
    assert sigma_pred.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect Sigma Pred shape: {sigma_pred.shape}"

    # Verify Sigma Positivity (Model uses Softplus)
    assert (sigma_pred > 0).all(), "Sigma predictions must be positive"

    print("Verification Successful: Data pipeline and Model forward pass are correct.")

    # 4. Training Loop
    print("\n[Step 3] Starting Training Loop (Debug Mode)...")
    # train_model handles the loop, validation, and saving the best checkpoint
    train_model(train_loader, val_loader)

    # Check if model was saved
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"
        )
    print(f"Model successfully saved to: {Config.BEST_MODEL_PATH}")

    # 5. Prediction / Inference
    print("\n[Step 4] Generating Predictions on Test Set...")
    predict(test_loader)

    # 6. Validate Submission File
    print("\n[Step 5] Validating Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in submission_df.columns for col in expected_cols
    ), f"Submission missing required columns. Found: {submission_df.columns}"

    # Check row count
    # In debug mode, the dataset __len__ is capped at 32.
    # However, the predict function iterates the DataLoader.
    # The submission file length should match the test dataset length.
    assert len(submission_df) == len(
        test_ds
    ), f"Submission row count ({len(submission_df)}) does not match Test Dataset ({len(test_ds)})"

    # Check Confidence Clipping (Metric requires max(sigma, 70))
    # The predict function applies this clipping.
    min_conf = submission_df["Confidence"].min()
    assert min_conf >= 70, f"Found confidence value < 70: {min_conf}"

    print("Submission File Stats:")
    print(submission_df.describe())

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
