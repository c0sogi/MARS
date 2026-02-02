import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import (
    DEVICE,
    IMG_SIZE,
    BATCH_SIZE,
    TARGET_MEAN,
    TARGET_STD,
    LR_BACKBONE,
    LR_HEAD,
    WEIGHT_DECAY,
)
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders, LungDataset
from library.model import RSTCNet
from library.train import LaplaceLogLikelihoodLoss, train_one_epoch, validate_one_epoch


def demo_data_pipeline():
    print("\n=== Demonstrating Data Pipeline ===")

    # Use debug=True to load a tiny subset (50 rows) for speed
    print("Loading dataloaders in debug mode...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch a single batch to verify shapes
    batch = next(iter(train_loader))
    images = batch["image"]
    tabular = batch["tabular"]
    time = batch["time"]
    targets = batch["target"]

    print(f"Batch keys: {batch.keys()}")
    print(f"Image shape: {images.shape} (Expected: [B, 3, {IMG_SIZE}, {IMG_SIZE}])")
    print(f"Tabular shape: {tabular.shape} (Expected: [B, 7])")
    print(f"Time shape: {time.shape} (Expected: [B, 1])")
    print(f"Target shape: {targets.shape} (Expected: [B, 1])")

    # Assertions
    assert images.ndim == 4, "Images should be 4D tensors (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels"
    assert tabular.shape[1] == 7, "Tabular data should have 7 features"
    assert time.shape[1] == 1, "Time should have 1 feature"
    assert targets.shape[1] == 1, "Target should have 1 feature"

    return train_loader, val_loader, batch


def demo_model_inference(batch):
    print("\n=== Demonstrating Model Inference ===")

    # Instantiate model
    model = RSTCNet(n_tabular_features=7).to(DEVICE)
    print("Model instantiated successfully.")

    # Prepare inputs
    images = batch["image"].to(DEVICE)
    tabular = batch["tabular"].to(DEVICE)
    time = batch["time"].to(DEVICE)

    # Forward pass
    model.eval()
    with torch.no_grad():
        mu, sigma = model(images, tabular, time)

    print(f"Prediction mu shape: {mu.shape}")
    print(f"Prediction sigma shape: {sigma.shape}")

    # Assertions
    assert mu.shape == (images.size(0), 1), "Mu output shape mismatch"
    assert sigma.shape == (images.size(0), 1), "Sigma output shape mismatch"
    assert (sigma > 0).all(), "Sigma must be positive (Softplus)"

    print("Model inference successful.")
    return model, mu, sigma


def demo_metric_and_loss(model, batch, mu, sigma):
    print("\n=== Demonstrating Metric and Loss ===")

    targets = batch["target"].to(DEVICE)

    # 1. Loss Calculation
    criterion = LaplaceLogLikelihoodLoss()
    loss = criterion(mu, sigma, targets)

    print(f"Calculated Loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "Loss should be finite"

    # 2. Metric Calculation
    # Convert tensors to numpy
    mu_np = mu.cpu().numpy()
    sigma_np = sigma.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # Inverse transform for metric calculation (as done in train.py)
    preds_mu_orig = mu_np * TARGET_STD + TARGET_MEAN
    preds_sigma_orig = sigma_np * TARGET_STD
    targets_orig = targets_np * TARGET_STD + TARGET_MEAN

    metric = calculate_metric(
        targets_orig.flatten(), preds_mu_orig.flatten(), preds_sigma_orig.flatten()
    )
    print(f"Calculated Metric: {metric:.4f}")
    assert np.isfinite(metric), "Metric should be finite"


def demo_training_loop(model, train_loader, val_loader):
    print("\n=== Demonstrating Training Loop (1 Epoch) ===")

    # Setup Optimizer
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params, "lr": LR_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    criterion = LaplaceLogLikelihoodLoss()

    # Run 1 Train Epoch
    print("Running training step...")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    print(f"Train Loss: {train_loss:.4f}")

    # Run 1 Validation Epoch
    print("Running validation step...")
    val_loss, val_metric = validate_one_epoch(model, val_loader, criterion, DEVICE)
    print(f"Val Loss: {val_loss:.4f} | Val Metric: {val_metric:.4f}")

    assert train_loss > -100, "Train loss seems suspiciously low/invalid"
    assert val_loss > -100, "Val loss seems suspiciously low/invalid"


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)

    try:
        # 1. Data
        train_loader, val_loader, sample_batch = demo_data_pipeline()

        # 2. Model
        model, mu, sigma = demo_model_inference(sample_batch)

        # 3. Metric/Loss
        demo_metric_and_loss(model, sample_batch, mu, sigma)

        # 4. Training Loop
        demo_training_loop(model, train_loader, val_loader)

        print("\nAll demonstrations completed successfully!")

    except AssertionError as e:
        print(f"\nAssertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Print full traceback for debugging if needed, but simple print is usually enough for this task
        import traceback

        traceback.print_exc()
        sys.exit(1)
