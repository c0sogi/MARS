import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

# Import library components
from library.config import Config
from library.utils import seed_everything, inverse_transform, laplace_log_likelihood
from library.data import get_train_val_datasets, get_test_dataset
from library.model import CCVRNet
from library.train import MetricAlignedLoss, train_one_epoch


def run_demo():
    print("=== Starting Library Usage Demo ===")

    # 1. Configuration Overrides for Speed
    # We modify the Config class attributes directly to run a fast demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # =========================================================================
    # 2. Data Pipeline Verification
    # =========================================================================
    print("\n[1/5] Verifying Data Pipeline...")

    # Load datasets
    train_ds, val_ds = get_train_val_datasets()

    # Create a small subset for testing (first 8 samples)
    subset_indices = list(range(8))
    train_subset = Subset(train_ds, subset_indices)

    # Check dataset item structure
    # Item: (image, tabular, target)
    sample_img, sample_tab, sample_target = train_subset[0]

    print(f"  Image Shape: {sample_img.shape}")
    print(f"  Tabular Shape: {sample_tab.shape}")
    print(f"  Target Shape: {sample_target.shape}")

    # Assertions for shapes
    # Image: (3 slices, 260, 260)
    assert sample_img.shape == (
        3,
        260,
        260,
    ), f"Expected (3, 260, 260), got {sample_img.shape}"
    # Tabular: 6 features (Baseline_FVC, Baseline_Percent, Age, Sex, Smoking, Relative_Weeks)
    assert sample_tab.shape == (6,), f"Expected (6,), got {sample_tab.shape}"
    # Target: 1 value (FVC_Scaled)
    assert sample_target.shape == (1,), f"Expected (1,), got {sample_target.shape}"

    # Verify DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Use 0 workers for simple debugging
    )

    batch_imgs, batch_tabs, batch_targets = next(iter(train_loader))
    print(f"  Batch Loaded. Images: {batch_imgs.shape}, Targets: {batch_targets.shape}")

    assert batch_imgs.size(0) == Config.BATCH_SIZE

    # =========================================================================
    # 3. Model Architecture Testing
    # =========================================================================
    print("\n[2/5] Verifying Model Architecture...")

    model = CCVRNet().to(device)

    # Move batch to device
    batch_imgs = batch_imgs.to(device)
    batch_tabs = batch_tabs.to(device)

    # Forward pass
    mu, sigma = model(batch_imgs, batch_tabs)

    print(f"  Output Mu Shape: {mu.shape}")
    print(f"  Output Sigma Shape: {sigma.shape}")

    # Assertions
    assert mu.shape == (Config.BATCH_SIZE,), "Mu shape mismatch"
    assert sigma.shape == (Config.BATCH_SIZE,), "Sigma shape mismatch"
    # Sigma must be positive (softplus + epsilon)
    assert (sigma > 0).all(), "Sigma contains non-positive values"

    # =========================================================================
    # 4. Loss and Metric Validation
    # =========================================================================
    print("\n[3/5] Verifying Loss and Metric...")

    criterion = MetricAlignedLoss()

    # Flatten targets to match output shape
    batch_targets_flat = batch_targets.view(-1).to(device)

    # Compute Loss
    loss = criterion(mu, sigma, batch_targets_flat)
    print(f"  Computed Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

    # Verify Metric Calculation (Inverse Transform + Laplace Log Likelihood)
    # Detach and move to CPU
    mu_np = mu.detach().cpu().numpy()
    sigma_np = sigma.detach().cpu().numpy()
    targets_np = batch_targets_flat.detach().cpu().numpy()

    # Inverse Transform
    mu_ml, sigma_ml = inverse_transform(mu_np, sigma_np)

    # Inverse transform targets manually for verification
    targets_ml = targets_np * Config.TARGET_STD + Config.TARGET_MEAN

    print(f"  Sample Prediction (ml): {mu_ml[0]:.2f} +/- {sigma_ml[0]:.2f}")
    print(f"  Sample Target (ml): {targets_ml[0]:.2f}")

    # Compute Metric
    metric_val = laplace_log_likelihood(targets_ml, mu_ml, sigma_ml)
    print(f"  Metric Score: {metric_val:.4f}")

    assert isinstance(metric_val, float)

    # =========================================================================
    # 5. Training Loop Demonstration
    # =========================================================================
    print("\n[4/5] Running Training Loop Demo (1 Epoch)...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run one epoch using the library function
    epoch_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"  Epoch Loss: {epoch_loss:.4f}")

    # =========================================================================
    # 6. Inference Workflow
    # =========================================================================
    print("\n[5/5] Verifying Inference Workflow...")

    # Load test dataset
    # We use the sample_submission.csv provided in input
    test_ds, sub_df = get_test_dataset()

    # Subset test dataset for speed (first 5)
    test_subset = Subset(test_ds, range(min(5, len(test_ds))))
    test_loader = DataLoader(test_subset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model.eval()
    predictions = []

    with torch.no_grad():
        for imgs, tabs, patient_ids in test_loader:
            imgs = imgs.to(device)
            tabs = tabs.to(device)

            mu_out, sigma_out = model(imgs, tabs)

            # Inverse transform
            mu_ml_out, sigma_ml_out = inverse_transform(
                mu_out.cpu().numpy(), sigma_out.cpu().numpy()
            )

            for i in range(len(patient_ids)):
                predictions.append({"FVC": mu_ml_out[i], "Confidence": sigma_ml_out[i]})

    print(f"  Generated {len(predictions)} predictions.")
    print("  First prediction:", predictions[0])

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
