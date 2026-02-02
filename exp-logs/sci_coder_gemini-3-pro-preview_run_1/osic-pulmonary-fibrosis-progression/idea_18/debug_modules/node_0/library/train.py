import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config, setup_reproducibility
from library.data import LungDataset, get_transforms
from library.model import CalibratedSymmetricDualAxisNetwork
from library.utils import calculate_metric


class LaplaceLoss(nn.Module):
    """
    Optimizes the Modified Laplace Log Likelihood directly.
    The objective is to minimize the Negative Log Likelihood, which corresponds
    to maximizing the competition metric.

    Formula:
    Loss = (sqrt(2) * Delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    where Delta = min(|True - Pred|, 1000)
    and sigma_clipped = max(sigma, 70)
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, sigma):
        # Flatten tensors
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        sigma = sigma.view(-1)

        # 1. Clip Confidence (sigma)
        # "confidence values are clipped at 70 ml"
        sigma_clipped = torch.clamp(sigma, min=Config.CONFIDENCE_CLIP)

        # 2. Calculate Absolute Error
        abs_error = torch.abs(y_true - y_pred)

        # 3. Clip Error (Delta)
        # "The error is thresholded at 1000 ml"
        # We use clamp to ensure gradients flow through the unclipped region
        delta = torch.clamp(abs_error, max=Config.ERR_CLIP_THRESHOLD)

        # 4. Compute Loss Terms
        sqrt_2 = np.sqrt(2)

        # Term 1: (sqrt(2) * delta) / sigma_clipped
        term1 = (sqrt_2 * delta) / sigma_clipped

        # Term 2: ln(sqrt(2) * sigma_clipped)
        term2 = torch.log(sqrt_2 * sigma_clipped)

        # Total Loss
        loss = term1 + term2

        return torch.mean(loss)


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move inputs to device
        imgs_ax = batch["image_axial"].to(device)
        imgs_cor = batch["image_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device)

        # Extract metadata for trajectory calculation
        weeks = batch["metadata"]["Weeks"].to(device)
        base_weeks = batch["metadata"]["Baseline_Week"].to(device)
        base_fvc = batch["metadata"]["Baseline_FVC"].to(device)

        optimizer.zero_grad()

        # Forward Pass: Predict parameters
        alpha, sigma_base, sigma_growth = model(imgs_ax, imgs_cor, tabular)

        # Calculate Trajectory Prediction
        # FVC_pred = Baseline_FVC + alpha * (Week - Baseline_Week)
        dt = weeks - base_weeks
        y_pred = base_fvc + alpha.view(-1) * dt

        # Calculate Confidence
        # Sigma = sigma_base + sigma_growth * |Week - Baseline_Week|
        sigma_total = sigma_base.view(-1) + sigma_growth.view(-1) * torch.abs(dt)

        # Compute Loss
        loss = criterion(targets, y_pred, sigma_total)

        # Backward Pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_epoch(model, loader, device):
    """
    Performs validation and calculates the competition metric.
    """
    model.eval()
    all_true = []
    all_pred = []
    all_sigma = []

    with torch.no_grad():
        for batch in loader:
            imgs_ax = batch["image_axial"].to(device)
            imgs_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)

            weeks = batch["metadata"]["Weeks"].to(device)
            base_weeks = batch["metadata"]["Baseline_Week"].to(device)
            base_fvc = batch["metadata"]["Baseline_FVC"].to(device)

            # Forward Pass
            alpha, sigma_base, sigma_growth = model(imgs_ax, imgs_cor, tabular)

            # Reconstruct Predictions
            dt = weeks - base_weeks
            y_pred = base_fvc + alpha.view(-1) * dt
            sigma_total = sigma_base.view(-1) + sigma_growth.view(-1) * torch.abs(dt)

            # Store for metric calculation
            all_true.extend(targets.cpu().numpy())
            all_pred.extend(y_pred.cpu().numpy())
            all_sigma.extend(sigma_total.cpu().numpy())

    # Calculate metric using the provided utility
    metric_score = calculate_metric(
        np.array(all_true), np.array(all_pred), np.array(all_sigma)
    )
    return metric_score


def run_training():
    """
    Main driver function for the training pipeline.
    """
    # 1. Setup
    setup_reproducibility(Config.SEED)
    Config.setup_directories()
    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # 2. Data Loading
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)

    if Config.DEBUG:
        print(f"DEBUG Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Initialize Datasets
    # Note: LungDataset handles image caching internally in 'cache_dir'
    train_dataset = LungDataset(
        train_df,
        cache_dir=Config.CACHE_DIR,
        transform=get_transforms("train"),
        mode="train",
    )

    val_dataset = LungDataset(
        val_df, cache_dir=Config.CACHE_DIR, transform=get_transforms("val"), mode="val"
    )

    # Initialize Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Calibrated Symmetric Dual-Axis Network...")
    model = CalibratedSymmetricDualAxisNetwork().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Loss Function
    criterion = LaplaceLoss()

    # 4. Training Loop
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Training for {Config.EPOCHS} epochs with patience {Config.PATIENCE}...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate_epoch(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Log Metrics (Full Precision for Val Score)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Metric: {val_score} | "
            f"Time: {elapsed:.1f}s"
        )

        # Early Stopping Logic
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Score! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
