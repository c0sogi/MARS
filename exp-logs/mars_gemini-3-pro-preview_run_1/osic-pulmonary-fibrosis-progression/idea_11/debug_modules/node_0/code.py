import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Paths, Training, System
from library.dataset import LungDataset
from library.model import PyramidDualAxisNet
from library.loss import LaplaceLogLikelihoodLoss
from library.engine import train_one_epoch, evaluate
from library.utils import seed_everything, laplace_log_likelihood


def main():
    print("Starting Lung Decline Prediction Pipeline Demo...")

    # 1. Setup
    seed_everything(Training.SEED)
    device = torch.device(System.DEVICE)
    print(f"Device: {device}")

    # 2. Data Loading (Training Subset)
    # We use a small subset to ensure the demo runs quickly
    print("\n--- Loading Training Data ---")
    if not os.path.exists(Paths.TRAIN_CSV):
        raise FileNotFoundError(f"Metadata file not found: {Paths.TRAIN_CSV}")

    df_train = pd.read_csv(Paths.TRAIN_CSV)
    # Select first 4 rows (likely belonging to the first patient or two)
    subset_train = df_train.head(4).copy()
    print(f"Selected training subset: {len(subset_train)} rows")

    # Instantiate Dataset
    # cache_images=False to force processing logic verification without relying on pre-existing cache
    train_ds = LungDataset(subset_train, mode="train", cache_images=False)

    # 3. Verify Dataset Item Structure
    print("\n--- Verifying Dataset Item ---")
    sample = train_ds[0]

    # Check keys
    expected_keys = [
        "axial",
        "coronal",
        "tabular",
        "time_delta",
        "baseline_fvc",
        "target",
        "patient_week",
    ]
    for k in expected_keys:
        assert k in sample, f"Missing key in dataset sample: {k}"

    # Check Shapes
    # Images: (3, 224, 224)
    assert sample["axial"].shape == (
        3,
        224,
        224,
    ), f"Incorrect Axial shape: {sample['axial'].shape}"
    assert sample["coronal"].shape == (
        3,
        224,
        224,
    ), f"Incorrect Coronal shape: {sample['coronal'].shape}"
    # Tabular: (4,) -> Age, Sex, Smoke, Percent
    assert sample["tabular"].shape == (
        4,
    ), f"Incorrect Tabular shape: {sample['tabular'].shape}"
    # Scalars
    assert sample["target"].shape == (
        1,
    ), f"Incorrect Target shape: {sample['target'].shape}"

    print("Dataset item structure verified successfully.")

    # 4. Model Initialization & Forward Pass
    print("\n--- Initializing Model ---")
    model = PyramidDualAxisNet().to(device)

    # Create DataLoader
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)

    # Get a batch
    batch = next(iter(train_loader))
    axial = batch["axial"].to(device)
    coronal = batch["coronal"].to(device)
    tabular = batch["tabular"].to(device)
    time_delta = batch["time_delta"].to(device)
    baseline_fvc = batch["baseline_fvc"].to(device)
    targets = batch["target"].to(device)

    print("Running forward pass...")
    pred_fvc, pred_sigma = model(axial, coronal, tabular, time_delta, baseline_fvc)

    # Verify Output Shapes
    assert pred_fvc.shape == (2, 1), f"Expected FVC shape (2, 1), got {pred_fvc.shape}"
    assert pred_sigma.shape == (
        2,
        1,
    ), f"Expected Sigma shape (2, 1), got {pred_sigma.shape}"

    # Verify Sigma Positivity (Softplus output should be > 0)
    assert (pred_sigma > 0).all(), "Predicted sigma contains non-positive values!"
    print("Forward pass successful. Output shapes verified.")

    # 5. Loss Function Verification
    print("\n--- Verifying Loss Function ---")
    criterion = LaplaceLogLikelihoodLoss()

    # Calculate Loss
    loss = criterion(pred_fvc, pred_sigma, targets)
    print(f"Calculated Loss: {loss.item():.4f}")

    # Verify against Metric Utility
    # Loss should be exactly negative of the metric
    metric_val = laplace_log_likelihood(targets, pred_fvc, pred_sigma)
    print(f"Calculated Metric: {metric_val:.4f}")

    # Check consistency (allow small float tolerance)
    assert np.isclose(
        loss.item(), -metric_val, atol=1e-5
    ), f"Mismatch: Loss {loss.item()} != -Metric {-metric_val}"
    print("Loss and Metric are consistent.")

    # 6. Training Loop Demo
    print("\n--- Running Training Loop (1 Epoch) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch Training Loss: {epoch_loss:.4f}")

    # 7. Evaluation Loop Demo
    print("\n--- Running Evaluation Loop ---")
    val_loss, val_metric = evaluate(model, train_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Metric: {val_metric:.4f}")

    # 8. Inference / Test Set Demo
    print("\n--- Verifying Test Inference ---")
    if os.path.exists(Paths.TEST_CSV):
        df_test = pd.read_csv(Paths.TEST_CSV)
        subset_test = df_test.head(2).copy()

        test_ds = LungDataset(subset_test, mode="test", cache_images=False)
        test_loader = DataLoader(test_ds, batch_size=2, shuffle=False)

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                # Move to device
                t_axial = batch["axial"].to(device)
                t_coronal = batch["coronal"].to(device)
                t_tabular = batch["tabular"].to(device)
                t_time_delta = batch["time_delta"].to(device)
                t_baseline_fvc = batch["baseline_fvc"].to(device)

                # Predict
                p_fvc, p_sigma = model(
                    t_axial, t_coronal, t_tabular, t_time_delta, t_baseline_fvc
                )

                # Check outputs
                print(
                    f"Test Batch Predictions - FVC Mean: {p_fvc.mean().item():.2f}, Sigma Mean: {p_sigma.mean().item():.2f}"
                )
                assert p_fvc.shape[0] == 2
                break
    else:
        print("Test metadata not found, skipping inference check.")

    print("\n" + "=" * 30)
    print("ALL CHECKS PASSED SUCCESSFULLY")
    print("=" * 30)


if __name__ == "__main__":
    main()
