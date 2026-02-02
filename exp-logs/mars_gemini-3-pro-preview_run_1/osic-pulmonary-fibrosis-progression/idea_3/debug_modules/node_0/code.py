import sys
import os
import torch
import pandas as pd
import numpy as np

# Import library modules
from library.config import Config
from library.data import get_dataloaders
from library.model import DualAxisTriSlabModel
from library.train import run_training, ParametricLoss, predict_trajectory
from library.utils import seed_everything


def main():
    print("=== Pulmonary Fibrosis Progression Prediction Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Override
    # ---------------------------------------------------------
    # We modify the Config class state to run a fast debug session.
    print("\n[1] Configuring environment...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 16  # Use only 16 samples for speed
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple demo

    # Redirect outputs to working directory
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Device: {Config.DEVICE}")

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # ---------------------------------------------------------
    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")
    # Initialize dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()

    # Fetch one batch from training set
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    axial = batch["axial"]
    coronal = batch["coronal"]
    tabular = batch["tabular"]
    targets = batch["target"]
    base_fvc = batch["base_fvc"]
    time_delta = batch["time_delta"]

    print(f"  Batch Size: {axial.size(0)}")
    print(f"  Axial Image Shape: {axial.shape}")  # Expected: (B, 3, 224, 224)
    print(f"  Coronal Image Shape: {coronal.shape}")  # Expected: (B, 3, 224, 224)
    print(f"  Tabular Shape: {tabular.shape}")  # Expected: (B, 5)
    print(f"  Targets Shape: {targets.shape}")  # Expected: (B,)

    # Assertions to ensure data integrity
    assert axial.shape == (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert coronal.shape == (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert tabular.shape == (Config.BATCH_SIZE, 5)
    assert targets.shape == (Config.BATCH_SIZE,)

    # ---------------------------------------------------------
    # 3. Model & Inference Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Model Architecture & Inference...")
    # Instantiate model
    # Note: We use pretrained=False here to ensure quick initialization for the check
    model = DualAxisTriSlabModel(
        backbone_name="efficientnet_b0",
        pretrained=False,
        tabular_input_dim=5,
        output_dim=3,
    )
    model.to(Config.DEVICE)
    model.eval()

    # Move batch to device
    axial_dev = axial.to(Config.DEVICE)
    coronal_dev = coronal.to(Config.DEVICE)
    tabular_dev = tabular.to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        outputs = model(axial_dev, coronal_dev, tabular_dev)

    print(f"  Model Output Shape: {outputs.shape}")  # Expected: (B, 3)

    # Assertions
    assert outputs.shape == (Config.BATCH_SIZE, 3)
    assert not torch.isnan(outputs).any(), "Model output contains NaNs"

    # ---------------------------------------------------------
    # 4. Loss & Metric Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Loss Calculation...")
    criterion = ParametricLoss()

    base_fvc_dev = base_fvc.to(Config.DEVICE)
    time_delta_dev = time_delta.to(Config.DEVICE)
    targets_dev = targets.to(Config.DEVICE)

    # Calculate loss
    loss = criterion(outputs, targets_dev, base_fvc_dev, time_delta_dev)
    print(f"  Loss Value: {loss.item():.4f}")

    # Verify prediction logic (FVC + Confidence)
    pred_fvc, pred_sigma = predict_trajectory(outputs, base_fvc_dev, time_delta_dev)

    print(f"  Sample Pred FVC: {pred_fvc[0].item():.2f}")
    print(f"  Sample Pred Sigma: {pred_sigma[0].item():.2f}")

    # Assertions
    assert loss.dim() == 0, "Loss must be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert (pred_sigma >= 70).all(), "Confidence values must be clipped to >= 70"

    # ---------------------------------------------------------
    # 5. Full Training Loop Execution
    # ---------------------------------------------------------
    print("\n[5] Executing Full Training Loop (via library.train.run_training)...")
    # This will use the overridden Config values (EPOCHS=2, DEBUG=True, etc.)
    # It handles training, validation, checkpointing, and submission generation.
    run_training()

    # ---------------------------------------------------------
    # 6. Submission Verification
    # ---------------------------------------------------------
    print("\n[6] Verifying Submission File...")
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission loaded. Shape: {sub_df.shape}")
    print(sub_df.head())

    # Assertions
    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}"

    # In Debug mode, the test set is also sliced to DEBUG_SAMPLES (16)
    assert (
        len(sub_df) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} rows, found {len(sub_df)}"

    assert not sub_df.isnull().values.any(), "Submission contains null values"
    assert (sub_df["Confidence"] >= 70).all(), "Confidence values in submission < 70"

    print("\n=== Demonstration Complete: Success ===")


if __name__ == "__main__":
    main()
