import os
import torch
import numpy as np
import pandas as pd
from torch import optim

# Import from the provided library
from library.config import Config
from library.utils import (
    seed_everything,
    laplace_log_likelihood_metric,
    inverse_scale_predictions,
)
from library.data import get_dataloaders
from library.model import TSCRNet
from library.train import LaplaceLogLikelihoodLoss, train_one_epoch, evaluate


def demo_pipeline():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration for demo...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config parameters for a fast demonstration
    Config.N_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SAMPLE_SIZE = 12  # Small subset for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {Config.DEVICE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Loading DataLoaders (Debug Mode)...")
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Verify DataLoader lengths
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch a single batch to inspect
    batch = next(iter(train_loader))

    images = batch["image"].to(Config.DEVICE)
    tabular = batch["tabular"].to(Config.DEVICE)
    time_abs = batch["time_abs"].to(Config.DEVICE)
    targets = batch["target"].to(Config.DEVICE)

    # Assertions for Data Shapes
    # Image: (B, 3, 260, 260) -> 3 slices, 260x260 resolution
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Image shape mismatch: {images.shape}"

    # Tabular: (B, 5) -> [BaseFVC, RelTime, Age, Sex, Smoke]
    assert tabular.shape == (
        Config.BATCH_SIZE,
        5,
    ), f"Tabular shape mismatch: {tabular.shape}"

    # Time Abs: (B, 1)
    assert time_abs.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Time Abs shape mismatch: {time_abs.shape}"

    # Target: (B, 1)
    assert targets.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Target shape mismatch: {targets.shape}"

    print("Data shapes verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Instantiation and Forward Pass
    # -------------------------------------------------------------------------
    print("\n>>> Instantiating TSCRNet...")
    model = TSCRNet().to(Config.DEVICE)

    print("Performing forward pass...")
    pred_mean, pred_sigma = model(images, tabular, time_abs)

    # Assertions for Model Output
    assert pred_mean.shape == (Config.BATCH_SIZE, 1), "Prediction Mean shape mismatch"
    assert pred_sigma.shape == (Config.BATCH_SIZE, 1), "Prediction Sigma shape mismatch"

    # Check positivity of sigma (Softplus + Epsilon ensures > 0)
    assert torch.all(pred_sigma > 0), "Sigma predictions must be positive"

    print(
        f"Forward pass successful. Mean: {pred_mean[0].item():.4f}, Sigma: {pred_sigma[0].item():.4f}"
    )

    # -------------------------------------------------------------------------
    # 4. Loss and Metric Calculation
    # -------------------------------------------------------------------------
    print("\n>>> Calculating Loss and Metric...")
    criterion = LaplaceLogLikelihoodLoss()

    # Compute Loss (on scaled data)
    loss = criterion(pred_mean, pred_sigma, targets)
    assert loss.dim() == 0, "Loss should be a scalar"
    print(f"Batch Loss: {loss.item():.4f}")

    # Compute Metric (on original scale)
    # 1. Inverse scale predictions
    pred_mean_orig, pred_sigma_orig = inverse_scale_predictions(
        pred_mean.detach().cpu().numpy(), pred_sigma.detach().cpu().numpy()
    )

    # 2. Inverse scale targets
    targets_orig = (
        targets.detach().cpu().numpy() * Config.TARGET_STD + Config.TARGET_MEAN
    )

    # 3. Calculate metric
    metric = laplace_log_likelihood_metric(
        targets_orig, pred_mean_orig, pred_sigma_orig
    )
    print(f"Batch Metric: {metric:.4f}")

    # -------------------------------------------------------------------------
    # 5. Training Loop Simulation
    # -------------------------------------------------------------------------
    print("\n>>> Running Short Training Simulation...")

    # Setup Optimizer
    backbone_params = list(map(id, model.backbone.parameters()))
    head_params = filter(lambda p: id(p) not in backbone_params, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEADS},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Run loop
    for epoch in range(Config.N_EPOCHS):
        print(f"--- Epoch {epoch + 1}/{Config.N_EPOCHS} ---")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )

        # Evaluate
        val_loss, val_metric = evaluate(model, val_loader, criterion, Config.DEVICE)

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Metric: {val_metric:.4f}"
        )

        # Basic sanity check: Loss should not be NaN
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"

    print("\n>>> Demo pipeline completed successfully.")


if __name__ == "__main__":
    demo_pipeline()
