import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library import utils, data, model, train


def run_demo():
    print("=== Starting Task Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Speed
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Initialize directories
    Config.setup()

    # Override Config defaults for speed and low resource usage
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for demo
    Config.BACKBONE_NAME = "tf_efficientnet_b0_ns"  # Smaller backbone for speed
    Config.IMG_SIZE = 224  # Slightly smaller image size

    # Set seed
    utils.seed_everything(Config.SEED)
    print("Configuration updated: EPOCHS=1, BATCH_SIZE=2, Backbone=EfficientNet-B0")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Loading Pipeline...")

    # Load dataloaders in debug mode (uses subset of data)
    train_loader, val_loader, sub_loader, sample_sub = data.get_dataloaders(debug=True)

    # Fetch one batch
    imgs, clin_data, targets = next(iter(train_loader))

    print(
        f"Batch shapes -> Images: {imgs.shape}, Clinical: {clin_data.shape}, Targets: {targets.shape}"
    )

    # Assertions
    # Images: (B, 3, H, W)
    assert imgs.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image tensor shape mismatch. Expected {(Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)}, got {imgs.shape}"

    # Clinical: (B, 9) -> [Base_FVC, Percent, Age, Rel_Time, Sex_M, Sex_F, Smoke_Ex, Smoke_Nev, Smoke_Cur]
    assert clin_data.shape == (
        Config.BATCH_SIZE,
        9,
    ), f"Clinical tensor shape mismatch. Expected {(Config.BATCH_SIZE, 9)}, got {clin_data.shape}"

    # Targets: (B, 1)
    assert targets.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Target tensor shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {targets.shape}"

    print("Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture and Inference...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.CLRNet().to(device)

    # Move batch to device
    imgs = imgs.to(device)
    clin_data = clin_data.to(device)

    # Forward pass
    preds = net(imgs, clin_data)

    print(f"Prediction shape: {preds.shape}")

    # Assertions
    # Output should be (B, 2) -> [Mean, Log_Sigma]
    assert preds.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 2)}, got {preds.shape}"

    print("Model inference verification passed.")

    # -------------------------------------------------------------------------
    # 4. Loss and Metric Verification
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Loss and Metric Calculation...")

    # Loss
    criterion = utils.LaplaceLogLikelihoodLoss()
    targets = targets.to(device)
    loss = criterion(preds, targets)

    print(f"Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.dim() == 0, "Loss should be a scalar"

    # Metric Utilities
    # Simulate predictions in standardized space
    mu_std = np.array([0.5, -0.2])  # Arbitrary Z-scores
    sigma_std = np.array([0.1, 0.2])  # Arbitrary scaled sigmas

    # Inverse Transform
    mu_ml, sigma_ml = utils.inverse_transform(mu_std, sigma_std)

    # Check logic: mu_ml = mu_std * TARGET_STD + TARGET_MEAN
    expected_mu = mu_std * Config.TARGET_STD + Config.TARGET_MEAN
    assert np.allclose(mu_ml, expected_mu), "Inverse transform for Mean incorrect"

    # Check logic: sigma_ml = sigma_std * TARGET_STD
    expected_sigma = sigma_std * Config.TARGET_STD
    assert np.allclose(
        sigma_ml, expected_sigma
    ), "Inverse transform for Sigma incorrect"

    # Calculate Metric
    # Case: Perfect prediction
    # delta = 0, sigma clipped at 70
    # metric = - ln(sqrt(2) * 70) approx -4.595
    y_true = np.array([2000, 3000])
    y_pred = np.array([2000, 3000])
    sigma_pred = np.array([10, 10])  # Will be clipped to 70

    metric_val = utils.calculate_metric(y_true, y_pred, sigma_pred)

    # Manual calc
    sigma_clipped = 70.0
    delta = 0.0
    expected_metric = -(np.sqrt(2) * delta / sigma_clipped) - np.log(
        np.sqrt(2) * sigma_clipped
    )

    print(
        f"Metric Check -> Calculated: {metric_val:.4f}, Expected: {expected_metric:.4f}"
    )
    assert np.isclose(
        metric_val, expected_metric, atol=1e-4
    ), "Metric calculation mismatch"

    print("Loss and Metric verification passed.")

    # -------------------------------------------------------------------------
    # 5. Full Training Pipeline Integration
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Full Training Loop (Short Duration)...")

    # We use the train module's run_training function
    # This will re-initialize data and model, but use our modified Config
    best_metric = train.run_training(debug=True)

    print(f"Training loop completed. Best Metric: {best_metric}")
    assert isinstance(best_metric, float), "run_training should return a float metric"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
