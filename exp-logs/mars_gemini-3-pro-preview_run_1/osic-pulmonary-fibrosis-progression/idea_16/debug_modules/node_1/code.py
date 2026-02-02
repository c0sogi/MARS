import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders
from library.model import VisuallyContextualizedNet
from library.train import train_model, LaplaceLogLikelihoodLoss


def run_demo():
    print("Initializing Demo/Verification Script...")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # ---------------------------------------------------------
    print("1. Overriding Configuration for Fast Execution...")

    # Enable Debug mode to use a small subset of data (50 train, 20 val/test)
    Config.DEBUG = True

    # Reduce training duration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid overhead for small demo

    # Redirect outputs to a demo-specific directory in working
    DEMO_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Ensure directories exist
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set Seed
    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("2. Verifying Data Pipeline (Loading & Processing)...")

    # This will trigger DICOM processing for the debug subset
    # We set load_cached_data=False to force the processing logic to run at least once
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))

    img_axial = batch["img_axial"]
    img_coronal = batch["img_coronal"]
    tabular = batch["tabular"]
    weeks = batch["weeks"]
    target = batch["target"]

    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Axial Image Shape: {img_axial.shape}")
    print(f"   Tabular Shape: {tabular.shape}")

    # Assertions
    assert img_axial.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Axial Image Shape: {img_axial.shape}"
    assert img_coronal.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Incorrect Coronal Image Shape: {img_coronal.shape}"
    assert tabular.shape == (
        Config.BATCH_SIZE,
        4,
    ), f"Incorrect Tabular Shape: {tabular.shape}"
    assert weeks.shape == (Config.BATCH_SIZE,), f"Incorrect Weeks Shape: {weeks.shape}"
    assert target.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect Target Shape: {target.shape}"

    print("   -> Data Pipeline Verified.")

    # ---------------------------------------------------------
    # 3. Model Architecture & Loss Verification
    # ---------------------------------------------------------
    print("3. Verifying Model Architecture and Loss...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisuallyContextualizedNet().to(device)
    criterion = LaplaceLogLikelihoodLoss()

    # Move batch to device
    img_axial = img_axial.to(device)
    img_coronal = img_coronal.to(device)
    tabular = tabular.to(device)
    weeks = weeks.to(device)
    target = target.to(device)

    # Forward Pass
    output = model(img_axial, img_coronal, tabular, weeks)

    fvc_pred = output["fvc"]
    confidence_pred = output["confidence"]

    # Assertions on Output
    assert fvc_pred.shape == (Config.BATCH_SIZE,), "Output FVC shape mismatch"
    assert confidence_pred.shape == (
        Config.BATCH_SIZE,
    ), "Output Confidence shape mismatch"
    assert not torch.isnan(fvc_pred).any(), "Model produced NaN FVC predictions"
    assert (confidence_pred >= 0).all(), "Confidence must be positive (Softplus)"

    # Loss Calculation
    loss = criterion(fvc_pred, confidence_pred, target)
    print(f"   Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"

    print("   -> Model and Loss Verified.")

    # ---------------------------------------------------------
    # 4. Integration Test: Full Training Loop
    # ---------------------------------------------------------
    print("4. Running Integration Test (Train Model Wrapper)...")

    # We use the library function train_model which handles the loop, validation, and submission
    # We pass load_cached_data=True now since we populated cache in step 2
    train_model(load_cached_data=True, epochs=1)

    # Check if best model was saved
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created."
    print("   -> Training Loop Completed and Model Saved.")

    # ---------------------------------------------------------
    # 5. Submission Verification
    # ---------------------------------------------------------
    print("5. Verifying Submission Output...")

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission Rows: {len(df_sub)}")
    print(f"   Columns: {list(df_sub.columns)}")

    # Assertions
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert all(
        col in df_sub.columns for col in expected_cols
    ), "Missing columns in submission."
    assert len(df_sub) > 0, "Submission file is empty."
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print("   -> Submission Verified.")

    print("\n" + "=" * 40)
    print("ALL CHECKS PASSED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    run_demo()
