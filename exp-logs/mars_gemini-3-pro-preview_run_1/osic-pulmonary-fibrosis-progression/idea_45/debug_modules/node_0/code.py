import os
import sys
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import SLHDAN, criterion
from library.train import run_training


def main():
    print(">>> 1. Setting up environment and seed...")
    seed_everything(Config.SEED)

    # Ensure working directory exists (Config creates it on import, but good to be safe)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(">>> 2. Verifying Data Pipeline (get_dataloaders)...")
    # Use debug=True to load a tiny subset (50 train, 20 val/test)
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))

    # Check keys
    required_keys = [
        "axial",
        "coronal",
        "tabular",
        "target",
        "dt",
        "baseline_fvc",
        "patient_week_id",
    ]
    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"

    # Check shapes
    # Images should be (B, 3, 224, 224)
    B = batch["axial"].shape[0]
    assert batch["axial"].shape == (
        B,
        3,
        224,
        224,
    ), f"Incorrect axial shape: {batch['axial'].shape}"
    assert batch["coronal"].shape == (
        B,
        3,
        224,
        224,
    ), f"Incorrect coronal shape: {batch['coronal'].shape}"

    # Tabular should be (B, 7) -> Weeks, Percent, Age, Sex, SmokingStatus(3)
    assert batch["tabular"].shape == (
        B,
        7,
    ), f"Incorrect tabular shape: {batch['tabular'].shape}"

    # Targets should be (B,)
    assert batch["target"].shape == (
        B,
    ), f"Incorrect target shape: {batch['target'].shape}"

    print(f"    Batch size: {B}")
    print("    Data pipeline verification passed.")

    print(">>> 3. Verifying Model Architecture (SLHDAN)...")
    device = Config.DEVICE
    model = SLHDAN().to(device)

    # Move batch to device
    axial = batch["axial"].to(device)
    coronal = batch["coronal"].to(device)
    tabular = batch["tabular"].to(device)
    target = batch["target"].to(device)
    dt = batch["dt"].to(device)
    baseline = batch["baseline_fvc"].to(device)

    # Forward pass
    alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

    # Check output shapes
    assert alpha.shape == (B,), f"Alpha shape mismatch: {alpha.shape}"
    assert sigma_base.shape == (B,), f"Sigma base shape mismatch: {sigma_base.shape}"
    assert sigma_growth.shape == (
        B,
    ), f"Sigma growth shape mismatch: {sigma_growth.shape}"

    # Test Criterion
    loss, fvc_pred, sigma_pred = criterion(
        alpha, sigma_base, sigma_growth, dt, baseline, target
    )

    assert loss.dim() == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert fvc_pred.shape == (B,), "Prediction shape mismatch"

    print("    Model forward pass and loss calculation passed.")

    print(">>> 4. Verifying Metric Logic...")
    # Test Case 1: Perfect prediction
    y_true = np.array([2000.0])
    y_pred_perf = np.array([2000.0])
    sigma_perf = np.array([70.0])  # Minimum sigma

    # Metric = - (sqrt(2)*0)/70 - ln(sqrt(2)*70) = -ln(98.99) approx -4.595
    metric_perf = laplace_log_likelihood_metric(y_true, y_pred_perf, sigma_perf)

    # Test Case 2: Bad prediction
    y_pred_bad = np.array([3000.0])  # Delta = 1000 (clipped)
    sigma_bad = np.array([70.0])

    # Metric = - (sqrt(2)*1000)/70 - ln(sqrt(2)*70) = -20.2 - 4.595 = -24.8 approx
    metric_bad = laplace_log_likelihood_metric(y_true, y_pred_bad, sigma_bad)

    assert (
        metric_perf > metric_bad
    ), "Perfect prediction should have higher metric than bad prediction"
    print(f"    Metric check passed. Perfect: {metric_perf:.4f}, Bad: {metric_bad:.4f}")

    print(">>> 5. Running Full Training Pipeline (Debug Mode)...")
    # This runs training, validation, and inference/submission generation
    # We set epochs=1 for speed
    run_training(debug=True, epochs=1)

    print(">>> 6. Verifying Submission File...")
    sub_path = Config.SUBMISSION_PATH

    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Submission file not found at {sub_path}")

    sub_df = pd.read_csv(sub_path)
    print(f"    Submission loaded. Shape: {sub_df.shape}")
    print(f"    Columns: {list(sub_df.columns)}")

    expected_cols = ["Patient_Week", "FVC", "Confidence"]
    for col in expected_cols:
        assert col in sub_df.columns, f"Missing column in submission: {col}"

    # Check if we have predictions
    assert len(sub_df) > 0, "Submission file is empty"
    assert not sub_df["FVC"].isnull().any(), "NaNs found in FVC predictions"
    assert (
        not sub_df["Confidence"].isnull().any()
    ), "NaNs found in Confidence predictions"

    print(">>> All demonstrations and verifications completed successfully.")


if __name__ == "__main__":
    main()
