import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, calculate_metric
from library.data import get_dataloaders
from library.model import NSLHN
from library.train import run_training
from library.predict import inference


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("=== Starting Demonstration of Lung Decline Prediction Pipeline ===\n")

    # 1. Setup and Configuration Overrides for Speed
    print("[Step 1] Setting up configuration and seeding...")
    seed_everything(42)

    # Override Config for rapid execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4  # Small batch size for demonstration
    Config.NUM_WORKERS = 2
    Config.setup()  # Create directories

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Device={Config.DEVICE}"
    )

    # 2. Data Loading Verification
    print("\n[Step 2] Verifying Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Fetch one batch to inspect
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # Verify keys
    expected_keys = [
        "patient_id",
        "image_axial",
        "image_coronal",
        "tabular",
        "target",
        "relative_week",
        "baseline_fvc",
    ]
    for key in expected_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Verify Shapes
    # Images: (B, 3, 224, 224)
    img_shape = (Config.BATCH_SIZE, 3, 224, 224)
    assert (
        batch["image_axial"].shape == img_shape
    ), f"Axial image shape mismatch: {batch['image_axial'].shape}"
    assert (
        batch["image_coronal"].shape == img_shape
    ), f"Coronal image shape mismatch: {batch['image_coronal'].shape}"

    # Tabular: (B, 6) -> [Percent, Age, Sex, Ex, Never, Current]
    assert batch["tabular"].shape == (
        Config.BATCH_SIZE,
        6,
    ), f"Tabular shape mismatch: {batch['tabular'].shape}"

    # Scalars: (B,)
    assert batch["relative_week"].shape == (
        Config.BATCH_SIZE,
    ), "Relative week shape mismatch"
    assert batch["baseline_fvc"].shape == (
        Config.BATCH_SIZE,
    ), "Baseline FVC shape mismatch"
    assert batch["target"].shape == (Config.BATCH_SIZE,), "Target shape mismatch"

    print("Data batch structure verified successfully.")

    # 3. Model Instantiation and Forward Pass
    print("\n[Step 3] Verifying Model Architecture and Forward Pass...")
    device = torch.device(Config.DEVICE)
    model = NSLHN().to(device)

    # Move batch to device
    img_ax = batch["image_axial"].to(device)
    img_cor = batch["image_coronal"].to(device)
    tabular = batch["tabular"].to(device)
    rel_week = batch["relative_week"].to(device)
    base_fvc = batch["baseline_fvc"].to(device)

    # Forward pass
    pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, rel_week, base_fvc)

    # Check outputs
    assert pred_fvc.shape == (
        Config.BATCH_SIZE,
    ), f"Pred FVC shape mismatch: {pred_fvc.shape}"
    assert pred_sigma.shape == (
        Config.BATCH_SIZE,
    ), f"Pred Sigma shape mismatch: {pred_sigma.shape}"

    # Check for NaNs
    assert not torch.isnan(pred_fvc).any(), "Model produced NaN in FVC prediction"
    assert not torch.isnan(pred_sigma).any(), "Model produced NaN in Sigma prediction"

    # Sigma must be positive (Softplus is used in model)
    assert (pred_sigma > 0).all(), "Model produced non-positive sigma values"

    print("Model forward pass successful.")

    # 4. Loss and Metric Calculation
    print("\n[Step 4] Verifying Loss and Metric Calculation...")
    criterion = LaplaceLogLikelihoodLoss(clip_sigma=70.0, clip_error=1000.0)
    target = batch["target"].to(device)

    # Calculate Loss
    loss = criterion(pred_fvc, pred_sigma, target)
    print(f"Calculated Loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss is not finite"

    # Calculate Metric
    metric = calculate_metric(
        pred_fvc.detach().cpu(), pred_sigma.detach().cpu(), target.cpu()
    )
    print(f"Calculated Metric: {metric:.4f}")
    assert np.isfinite(metric), "Metric is not finite"

    print("Loss and Metric functions verified.")

    # 5. Full Training Cycle Integration
    print("\n[Step 5] Running Full Training Cycle (1 Epoch)...")
    # This function handles training, validation, saving best model, and generating submission
    run_training(epochs=1)

    # Check if model checkpoint exists
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
    print("Training cycle completed and model saved.")

    # 6. Submission Verification
    print("\n[Step 6] Verifying Submission File...")
    # run_training calls generate_submission at the end, so file should exist
    assert os.path.exists(
        Config.SUBMISSION_FILE
    ), f"Submission file not found at {Config.SUBMISSION_FILE}"

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission shape: {sub_df.shape}")

    # Verify Columns
    required_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in required_cols:
        assert col in sub_df.columns, f"Missing column in submission: {col}"

    # Verify content validity
    assert not sub_df["FVC"].isnull().any(), "Submission contains null FVCs"
    assert (
        not sub_df["Confidence"].isnull().any()
    ), "Submission contains null Confidences"
    assert (
        sub_df["Confidence"] >= 70
    ).all(), "Confidence values below clipped threshold (70)"

    print("Submission file verified successfully.")

    # Optional: Explicit Inference Call check (if one wanted to run inference separately)
    print("\n[Optional] Running explicit inference call...")
    inference(
        model_path=Config.MODEL_SAVE_PATH, batch_size=Config.BATCH_SIZE, debug=True
    )

    print("\n=== Demonstration Complete: All checks passed. ===")


if __name__ == "__main__":
    main()
