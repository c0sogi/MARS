import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config, seed_everything
from library.data import get_dataloaders, get_test_dataloader
from library.model import DBSLNet
from library.train import CustomLoss, run_training
from library.predict import run_prediction
from library.utils import metric_laplace_log_likelihood


def main():
    print("Initializing Demonstration...")

    # 1. Setup and Configuration Overrides for Speed
    # We override Config attributes to ensure the demo runs quickly within the time limit.
    seed_everything(42)

    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4  # Small batch size for demo
    Config.INFERENCE_BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    # Ensure working directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n[1] Testing Data Loading (Debug Mode)...")

    # Use debug=True to load only a small subset (50 samples) of the training data
    train_loader, val_loader = get_dataloaders(debug=True)

    print(f"Train Loader Length (batches): {len(train_loader)}")
    print(f"Val Loader Length (batches): {len(val_loader)}")

    # Fetch one batch to verify structure and shapes
    batch = next(iter(train_loader))
    required_keys = [
        "image_axial",
        "image_coronal",
        "tabular",
        "target",
        "week",
        "base_week",
        "base_fvc",
        "patient_id",
    ]

    # Check for existence of keys
    for key in required_keys:
        if key not in batch:
            raise AssertionError(f"Missing key in batch: {key}")

    # Verify Tensor Shapes
    # Images should be (Batch, 3, 224, 224)
    assert batch["image_axial"].shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Incorrect Axial Image Shape: {batch['image_axial'].shape}"
    assert batch["image_coronal"].shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Incorrect Coronal Image Shape: {batch['image_coronal'].shape}"

    # Tabular data should be (Batch, 4) -> [Age, Percent, Sex, Smoke]
    assert batch["tabular"].shape == (
        Config.BATCH_SIZE,
        4,
    ), f"Incorrect Tabular Shape: {batch['tabular'].shape}"

    print("Data Loading Verified.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[2] Testing Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    model = DBSLNet().to(device)

    # Move batch data to device
    img_ax = batch["image_axial"].to(device)
    img_cor = batch["image_coronal"].to(device)
    tabular = batch["tabular"].to(device)
    week = batch["week"].to(device)
    base_week = batch["base_week"].to(device)
    base_fvc = batch["base_fvc"].to(device)

    # Perform Forward Pass
    # The model returns (FVC_Pred, Sigma_Pred)
    fvc_pred, sigma_pred = model(img_ax, img_cor, tabular, week, base_week, base_fvc)

    # Verify output shapes: Should be (Batch_Size,)
    assert fvc_pred.shape == (
        Config.BATCH_SIZE,
    ), f"FVC Pred shape mismatch: {fvc_pred.shape}"
    assert sigma_pred.shape == (
        Config.BATCH_SIZE,
    ), f"Sigma Pred shape mismatch: {sigma_pred.shape}"

    print("Model Forward Pass Verified.")

    # ==========================================
    # 4. Loss Function Verification
    # ==========================================
    print("\n[3] Testing Loss Function...")
    criterion = CustomLoss().to(device)
    target = batch["target"].to(device)

    # Calculate loss
    loss = criterion(fvc_pred, sigma_pred, target)

    # Verify loss is a valid scalar
    assert loss.dim() == 0, "Loss should be a scalar tensor"
    assert not torch.isnan(loss), "Loss is NaN"

    print(f"Loss Calculated: {loss.item():.4f}")

    # ==========================================
    # 5. Training Loop Execution
    # ==========================================
    print("\n[4] Testing Training Loop (1 Epoch)...")
    # run_training handles the loop, validation, optimizer, and checkpoint saving.
    # We pass debug=True to use the small subset and epochs=1 for speed.
    run_training(debug=True, epochs=1)

    # Verify that the model checkpoint was created
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
        )

    print("Training Loop Verified.")

    # ==========================================
    # 6. Inference and Submission
    # ==========================================
    print("\n[5] Testing Inference and Submission...")
    # run_prediction loads the saved model, runs inference on the test set,
    # and generates the submission CSV.
    run_prediction()

    # Verify submission file existence
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Verify submission content format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")

    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    if list(sub_df.columns) != expected_cols:
        raise ValueError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"
        )

    # Check for NaNs
    if sub_df.isnull().values.any():
        raise ValueError("Submission contains NaN values.")

    # Check Confidence clipping logic (Metric requires min 70)
    min_conf = sub_df["Confidence"].min()
    if min_conf < 70:
        raise ValueError(
            f"Confidence values not clipped correctly. Min found: {min_conf}"
        )

    print("Inference and Submission Verified.")

    # ==========================================
    # 7. Metric Utility Verification
    # ==========================================
    print("\n[6] Testing Metric Utility...")
    # Test metric with a perfect prediction scenario
    y_true = np.array([2000.0, 3000.0])
    y_pred = np.array([2000.0, 3000.0])
    sigma = np.array([100.0, 100.0])

    # Formula: - (sqrt(2)*delta)/sigma - ln(sqrt(2)*sigma)
    # With delta=0: - ln(sqrt(2)*100) = -ln(141.421356) ≈ -4.9517
    score = metric_laplace_log_likelihood(y_true, y_pred, sigma)
    expected_score = -np.log(np.sqrt(2) * 100)

    if not np.isclose(score, expected_score, atol=1e-4):
        raise ValueError(
            f"Metric calculation mismatch. Got {score}, expected {expected_score}"
        )

    print(f"Metric Verified. Score for perfect prediction with sigma=100: {score:.4f}")

    print("\n" + "=" * 40)
    print("ALL DEMONSTRATION STEPS COMPLETED SUCCESSFULLY")
    print("=" * 40)


if __name__ == "__main__":
    main()
