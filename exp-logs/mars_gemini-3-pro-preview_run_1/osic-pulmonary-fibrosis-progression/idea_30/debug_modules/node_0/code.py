import os
import torch
import pandas as pd
import numpy as np
import sys

# Import from the provided library files
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import CVERNet, LaplaceLikelihoodLoss, predict_and_submit
from library.train import train_one_epoch, validate
from library.utils import score_function


def main():
    print("=" * 40)
    print("C-VER-Net Library Usage Demonstration")
    print("=" * 40)

    # ---------------------------------------------------------
    # 1. Configuration Setup
    # ---------------------------------------------------------
    print("\n[1] Setting up Configuration...")

    # Override Config for a fast demonstration (Debug Mode)
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 20  # Use only 20 samples
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution

    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"    Device: {device}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")

    # ---------------------------------------------------------
    # 2. Data Loading
    # ---------------------------------------------------------
    print("\n[2] Initializing Data Loaders...")

    # Get dataloaders with the overridden config
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Fetch a single batch to verify structure
    print("    Verifying batch structure...")
    batch = next(iter(train_loader))

    # Check for required keys
    required_keys = [
        "img_axial",
        "img_coronal",
        "tabular",
        "target",
        "weeks",
        "base_fvc",
        "base_week",
        "patient_id",
    ]
    for key in required_keys:
        if key not in batch:
            raise AssertionError(f"Missing key in batch: {key}")

    # Verify Image Shapes: (Batch, 3, 224, 224)
    img_shape = batch["img_axial"].shape
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
    if img_shape != expected_img_shape:
        raise AssertionError(
            f"Image shape mismatch. Expected {expected_img_shape}, got {img_shape}"
        )

    # Verify Tabular Shapes: (Batch, 7)
    # 7 features: Age(1) + Percent(1) + Sex(2) + Smoking(3)
    tab_shape = batch["tabular"].shape
    if tab_shape[1] != 7:
        raise AssertionError(
            f"Tabular feature dim mismatch. Expected 7, got {tab_shape[1]}"
        )

    print("    Batch verification successful.")

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[3] Initializing Model and Running Forward Pass...")

    model = CVERNet().to(device)

    # Prepare inputs from the batch
    img_ax = batch["img_axial"].to(device)
    img_cor = batch["img_coronal"].to(device)
    tabular = batch["tabular"].to(device)
    weeks = batch["weeks"].to(device)
    base_fvc = batch["base_fvc"].to(device)
    base_week = batch["base_week"].to(device)
    target = batch["target"].to(device)

    # Forward pass
    fvc_pred, sigma_pred = model(img_ax, img_cor, tabular, weeks, base_fvc, base_week)

    print(f"    Predictions Shape: {fvc_pred.shape}")
    print(f"    Confidence Shape:  {sigma_pred.shape}")

    # Assertions
    if fvc_pred.shape != (Config.BATCH_SIZE,):
        raise AssertionError("FVC prediction shape mismatch")
    if sigma_pred.shape != (Config.BATCH_SIZE,):
        raise AssertionError("Sigma prediction shape mismatch")
    if torch.any(sigma_pred <= 0):
        raise AssertionError("Confidence (sigma) must be positive")

    # ---------------------------------------------------------
    # 4. Loss Calculation
    # ---------------------------------------------------------
    print("\n[4] calculating Loss...")

    loss_fn = LaplaceLikelihoodLoss()
    loss = loss_fn(target, fvc_pred, sigma_pred, device)

    print(f"    Loss Value: {loss.item():.4f}")

    if torch.isnan(loss):
        raise AssertionError("Loss is NaN")

    # ---------------------------------------------------------
    # 5. Training Loop Simulation
    # ---------------------------------------------------------
    print("\n[5] Simulating Training Loop (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Train for one epoch
    avg_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
    print(f"    Epoch Training Loss: {avg_loss:.4f}")

    # Validate
    val_score = validate(model, val_loader, device)
    print(f"    Validation Score:    {val_score:.4f}")

    # ---------------------------------------------------------
    # 6. Inference & Submission
    # ---------------------------------------------------------
    print("\n[6] Generating Submission...")

    # Save the current model as 'best_model.pth' so inference can load it
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"    Saved dummy best model to {Config.BEST_MODEL_PATH}")

    # Run inference pipeline
    predict_and_submit(test_loader)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise AssertionError(f"Submission file not found at {Config.SUBMISSION_FILE}")

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission generated with {len(sub_df)} rows.")
    print(f"    Columns: {list(sub_df.columns)}")

    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    if list(sub_df.columns) != expected_cols:
        raise AssertionError(f"Submission columns mismatch. Expected {expected_cols}")

    print("\n" + "=" * 40)
    print("Demonstration Completed Successfully!")
    print("=" * 40)


if __name__ == "__main__":
    main()
