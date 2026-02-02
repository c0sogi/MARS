import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, LaplaceLogLikelihoodLoss
from library.data import get_dataloaders
from library.model import HiFiDACR
from library.train import run_training


def main():
    print(
        "=== Starting Demonstration of Pulmonary Fibrosis Progression Prediction Pipeline ===\n"
    )

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Demo
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for fast demonstration...")

    # Define demo-specific directories in ./working
    DEMO_WORK_DIR = "./working/demo_execution"
    DEMO_CACHE_DIR = "./working/demo_cache"

    os.makedirs(DEMO_WORK_DIR, exist_ok=True)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    # Override Config attributes to isolate this run and speed it up
    Config.OUTPUT_DIR = DEMO_WORK_DIR
    Config.CACHE_DIR = DEMO_CACHE_DIR
    Config.BEST_MODEL_PATH = os.path.join(DEMO_WORK_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_WORK_DIR, "demo_submission.csv")

    # Optimization for speed
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEBUG = True  # Use subset of data

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print(f"    Output Directory: {Config.OUTPUT_DIR}")
    print(f"    Cache Directory:  {Config.CACHE_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Initialize DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.DEBUG)

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Extract components
    imgs_ax = batch["image_axial"]
    imgs_cor = batch["image_coronal"]
    tabular = batch["tabular"]
    targets = batch["target"]
    base_fvc = batch["baseline_fvc"]
    rel_week = batch["relative_week"]

    # Assertions
    print("    Verifying batch shapes and types...")
    assert imgs_ax.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Axial Image Shape: {imgs_ax.shape}"
    assert imgs_cor.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Coronal Image Shape: {imgs_cor.shape}"
    assert (
        tabular.ndim == 2 and tabular.shape[0] == Config.BATCH_SIZE
    ), f"Incorrect Tabular Shape: {tabular.shape}"
    assert targets.shape[0] == Config.BATCH_SIZE, "Incorrect Target Shape"

    # Verify value ranges (Images should be normalized [0, 1])
    assert (
        imgs_ax.max() <= 1.0 and imgs_ax.min() >= 0.0
    ), "Axial images not normalized to [0, 1]"
    assert (
        imgs_cor.max() <= 1.0 and imgs_cor.min() >= 0.0
    ), "Coronal images not normalized to [0, 1]"

    print("    Data Pipeline Verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model & Loss Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture and Loss Function...")

    device = Config.DEVICE
    tab_dim = tabular.shape[1]

    # Instantiate Model
    model = HiFiDACR(tab_input_dim=tab_dim).to(device)

    # Move batch to device
    imgs_ax = imgs_ax.to(device)
    imgs_cor = imgs_cor.to(device)
    tabular = tabular.to(device)
    base_fvc = base_fvc.to(device)
    rel_week = rel_week.to(device)
    targets = targets.to(device)

    # Forward Pass
    outputs = model(
        image_axial=imgs_ax,
        image_coronal=imgs_cor,
        tabular=tabular,
        baseline_fvc=base_fvc,
        relative_week=rel_week,
    )

    # Verify Output Shape (Batch, 2) -> [FVC, Confidence]
    assert outputs.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Model output shape mismatch: {outputs.shape}"

    # Verify Confidence is positive (Softplus constraint)
    confidence = outputs[:, 1]
    assert torch.all(confidence > 0), "Model predicted non-positive confidence values."

    # Loss Calculation
    criterion = LaplaceLogLikelihoodLoss().to(device)
    loss = criterion(outputs, targets)

    # Verify Loss
    assert loss.dim() == 0, "Loss should be a scalar."
    assert not torch.isnan(loss), "Loss is NaN."

    print(f"    Forward pass successful. Loss: {loss.item():.4f}")
    print("    Model and Loss Verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (1 Epoch)...")

    # We use the provided run_training function which encapsulates the loop
    # We pass num_epochs=1 explicitly
    run_training(debug=True, num_epochs=1)

    # Verify artifact generation
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"    Training complete. Model saved to {Config.BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError(
            f"Training failed to save model at {Config.BEST_MODEL_PATH}"
        )

    # -------------------------------------------------------------------------
    # 5. Inference & Submission Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Inference and Submission Generation...")

    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    predictions = []

    print("    Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tab = batch["tabular"].to(device)
            b_fvc = batch["baseline_fvc"].to(device)
            r_week = batch["relative_week"].to(device)
            p_weeks = batch["patient_week"]  # List of IDs

            out = model(
                image_axial=img_ax,
                image_coronal=img_cor,
                tabular=tab,
                baseline_fvc=b_fvc,
                relative_week=r_week,
            )

            fvc_pred = out[:, 0].cpu().numpy()
            conf_pred = out[:, 1].cpu().numpy()

            for i in range(len(p_weeks)):
                predictions.append(
                    {
                        "Patient_Week": p_weeks[i],
                        "FVC": fvc_pred[i],
                        "Confidence": conf_pred[i],
                    }
                )

    # Create Submission DataFrame
    sub_df = pd.DataFrame(predictions)

    # Assertions on Submission
    assert "Patient_Week" in sub_df.columns
    assert "FVC" in sub_df.columns
    assert "Confidence" in sub_df.columns
    assert len(sub_df) > 0, "No predictions generated."

    # Save submission
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")
    print(f"    Generated {len(sub_df)} predictions.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
