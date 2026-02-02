import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_loss
from library.data import get_dataloaders
from library.model import SLHDANetwork
from library.train import train_one_epoch, validate
from library.inference import predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print(">>> Starting SLH-DAN Pipeline Demonstration...")

    # 1. Configuration Overrides for Speed
    # We modify the global Config class directly to ensure fast execution
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small subset for demo
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directories exist
    Config.setup()

    # Set reproducible seed
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")
    print("Configuration updated for fast demonstration.")

    # 2. Data Loading & Verification
    print("\n>>> 2. Verifying Data Pipeline...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch one batch
    batch = next(iter(train_loader))

    # Define expected shapes
    B = Config.BATCH_SIZE
    IMG_SIZE = Config.IMG_SIZE
    TAB_DIM = 6  # Age, Sex, Smoke_Ex, Smoke_Never, Smoke_Current, Percent

    # Extract data
    img_ax = batch["img_axial"]
    img_cor = batch["img_coronal"]
    tab = batch["tabular"]
    target = batch["target"]
    weeks = batch["weeks"]
    base_fvc = batch["base_fvc"]

    # Assertions
    print(f"Checking batch shapes (Batch Size: {B})...")

    # Image shapes: (B, 3, 224, 224)
    assert img_ax.shape == (
        B,
        3,
        IMG_SIZE,
        IMG_SIZE,
    ), f"Axial image shape mismatch. Expected {(B, 3, IMG_SIZE, IMG_SIZE)}, got {img_ax.shape}"
    assert img_cor.shape == (
        B,
        3,
        IMG_SIZE,
        IMG_SIZE,
    ), f"Coronal image shape mismatch. Expected {(B, 3, IMG_SIZE, IMG_SIZE)}, got {img_cor.shape}"

    # Tabular shape: (B, 6)
    assert tab.shape == (
        B,
        TAB_DIM,
    ), f"Tabular data shape mismatch. Expected {(B, TAB_DIM)}, got {tab.shape}"

    # Target and Meta shapes: (B,)
    assert target.shape == (
        B,
    ), f"Target shape mismatch. Expected {(B,)}, got {target.shape}"
    assert weeks.shape == (
        B,
    ), f"Weeks shape mismatch. Expected {(B,)}, got {weeks.shape}"
    assert base_fvc.shape == (
        B,
    ), f"Base FVC shape mismatch. Expected {(B,)}, got {base_fvc.shape}"

    print("Data pipeline verification passed.")

    # 3. Model Initialization & Forward Pass
    print("\n>>> 3. Verifying Model Architecture...")
    model = SLHDANetwork().to(device)

    # Move batch to device
    img_ax = img_ax.to(device)
    img_cor = img_cor.to(device)
    tab = tab.to(device)
    weeks = weeks.to(device)
    base_fvc = base_fvc.to(device)
    base_week = batch["base_week"].to(device)
    target = target.to(device)

    # Forward pass
    pred_fvc, pred_sigma = model(img_ax, img_cor, tab, weeks, base_fvc, base_week)

    # Verify outputs
    assert pred_fvc.shape == (B,), f"Pred FVC shape mismatch. Got {pred_fvc.shape}"
    assert pred_sigma.shape == (
        B,
    ), f"Pred Sigma shape mismatch. Got {pred_sigma.shape}"

    # Check for NaNs
    if torch.isnan(pred_fvc).any() or torch.isnan(pred_sigma).any():
        raise AssertionError("Model produced NaN predictions in forward pass.")

    print(f"Forward pass successful. Output shapes: {pred_fvc.shape}")

    # 4. Loss Calculation
    print("\n>>> 4. Verifying Loss Function...")
    loss = laplace_log_likelihood_loss(target, pred_fvc, pred_sigma)

    assert loss.dim() == 0, "Loss should be a scalar."
    assert not torch.isnan(loss), "Loss is NaN."
    assert not torch.isinf(loss), "Loss is Infinite."

    print(f"Loss calculation successful. Loss value: {loss.item():.4f}")

    # 5. Training Loop Simulation
    print("\n>>> 5. Simulating Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    # Train one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Training epoch complete. Avg Loss: {train_loss:.4f}")

    # Validate
    val_metric = validate(model, val_loader, device)
    print(f"Validation complete. Metric: {val_metric:.4f}")

    # Save model for inference step
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"Model checkpoint saved to {Config.MODEL_SAVE_PATH}")

    # 6. Inference & Submission
    print("\n>>> 6. Running Inference & Generating Submission...")

    # We use the provided inference function
    # It will load the model we just saved
    predict_and_submit(
        model_path=Config.MODEL_SAVE_PATH,
        output_path=Config.SUBMISSION_PATH,
        device=Config.DEVICE,
        debug=True,
    )

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {sub_df.shape}")

    # Check columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    if list(sub_df.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"
        )

    # Check for empty file
    if len(sub_df) == 0:
        raise AssertionError("Submission file is empty.")

    # Check values
    if sub_df["FVC"].isnull().any() or sub_df["Confidence"].isnull().any():
        raise AssertionError("Submission contains Null values.")

    print("Submission file verification passed.")
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
