import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders, get_test_loader
from library.model import PGBBNet
from library.loss import LaplaceNLLLoss
from library.train import train_model
from library.inference import generate_submission


def run_demo():
    print("=" * 50)
    print("Starting PGBB-Net Pipeline Demonstration")
    print("=" * 50)

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config parameters to run a minimal version
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.DEBUG = True  # Enable debug mode
    Config.DEBUG_SAMPLE_SIZE = 16  # Small subset of data
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    print("Configuration updated: 1 Epoch, Batch Size 4, Debug Mode ON.")

    # ---------------------------------------------------------
    # 2. Data Loading Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(debug=True)

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = ["axial", "coronal", "tabular", "meta", "target"]
    assert all(
        k in batch for k in expected_keys
    ), f"Missing keys in batch. Found: {batch.keys()}"

    # Verify shapes
    # Axial/Coronal: (B, 3, 224, 224)
    B = Config.BATCH_SIZE
    assert batch["axial"].shape == (
        B,
        3,
        224,
        224,
    ), f"Incorrect Axial shape: {batch['axial'].shape}"
    assert batch["coronal"].shape == (
        B,
        3,
        224,
        224,
    ), f"Incorrect Coronal shape: {batch['coronal'].shape}"

    # Tabular: (B, 4) -> [Age, Sex, Smoke, Percent]
    assert batch["tabular"].shape == (
        B,
        4,
    ), f"Incorrect Tabular shape: {batch['tabular'].shape}"

    # Meta: (B, 2) -> [rel_week, base_fvc]
    assert batch["meta"].shape == (B, 2), f"Incorrect Meta shape: {batch['meta'].shape}"

    # Target: (B,)
    assert batch["target"].shape == (
        B,
    ), f"Incorrect Target shape: {batch['target'].shape}"

    print("Data Loader verification passed. Batch shapes are correct.")

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = PGBBNet().to(device)

    # Move batch to device
    axial = batch["axial"].to(device)
    coronal = batch["coronal"].to(device)
    tabular = batch["tabular"].to(device)
    meta = batch["meta"].to(device)

    # Forward pass
    preds = model(axial, coronal, tabular, meta)

    # Verify output shape: (B, 2) -> [FVC, Confidence]
    assert preds.shape == (B, 2), f"Incorrect Model Output shape: {preds.shape}"

    # Verify values are finite (no NaNs)
    assert torch.isfinite(preds).all(), "Model output contains NaNs or Infs."

    print("Model forward pass successful. Output shape: (Batch, 2).")

    # ---------------------------------------------------------
    # 4. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Laplace Log Likelihood Loss...")

    criterion = LaplaceNLLLoss().to(device)
    target = batch["target"].to(device)

    loss = criterion(preds, target)

    # Verify loss is a scalar
    assert loss.dim() == 0, f"Loss should be scalar, got dim {loss.dim()}"
    assert torch.isfinite(loss), "Loss value is not finite."

    print(f"Loss calculation successful. Loss value: {loss.item():.4f}")

    # ---------------------------------------------------------
    # 5. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch)...")

    # Run the training routine provided in library/train.py
    # This will save checkpoints to Config.WORKING_DIR
    best_val_loss = train_model(debug=True)

    print(f"Training finished. Best Validation Loss: {best_val_loss:.4f}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "best_model.pth was not created."
    print("Checkpoint file verified.")

    # ---------------------------------------------------------
    # 6. Inference & Submission Verification
    # ---------------------------------------------------------
    print("\n[6] Running Inference and Generating Submission...")

    submission_file = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Generate submission using the trained model
    generate_submission(checkpoint_path=checkpoint_path, output_file=submission_file)

    # Verify Submission File
    assert os.path.exists(submission_file), "Submission file was not created."

    df_sub = pd.read_csv(submission_file)

    # Check Columns
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert list(df_sub.columns) == expected_cols, f"Incorrect columns: {df_sub.columns}"

    # Check Row Count (Test set size)
    # Note: In debug mode, get_test_loader loads the full test set unless modified,
    # but the provided get_test_loader doesn't take a debug flag.
    # The test.csv in metadata has 1908 rows.
    print(f"Submission rows: {len(df_sub)}")
    assert len(df_sub) > 0, "Submission file is empty."

    # Check Confidence Clipping (Metric requires min 70)
    min_conf = df_sub["Confidence"].min()
    assert (
        min_conf >= 70.0
    ), f"Confidence clipping failed. Min confidence found: {min_conf}"

    print("Submission verification passed.")

    print("\n" + "=" * 50)
    print("ALL DEMONSTRATION STEPS COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
