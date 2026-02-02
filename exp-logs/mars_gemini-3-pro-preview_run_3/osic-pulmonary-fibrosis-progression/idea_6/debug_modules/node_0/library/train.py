import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.data import (
    CTScanProcessor,
    TabularScaler,
    PulmonaryDataset,
    get_baseline_lookup,
)
from library.model import CAPNet
from library.utils import AverageMeter, metric_laplace_log_likelihood


def laplace_log_likelihood_loss(mu, sigma, target):
    """
    Computes the Negative Laplace Log Likelihood loss.
    L = (sqrt(2) * |y - mu|) / sigma + log(sqrt(2) * sigma)

    This is computed on scaled values for numerical stability during training.
    """
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=mu.device))

    # Add a small epsilon to sigma inside log just in case, though softplus+1e-3 in model handles it
    sigma = torch.max(sigma, torch.tensor(1e-6, device=mu.device))

    abs_diff = torch.abs(target - mu)
    loss = (sqrt_2 * abs_diff) / sigma + torch.log(sqrt_2 * sigma)
    return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, device):
    """
    Handles the training of one epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        # Move inputs to device
        image = batch["image"].to(device)
        meta_cat = batch["meta_cat"].to(device)
        meta_num = batch["meta_num"].to(device)
        baseline_fvc_scaled = batch["baseline_fvc_scaled"].to(device)
        weeks_scaled = batch["weeks_scaled"].to(device)
        target_fvc_scaled = batch["target_fvc_scaled"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        mu, sigma = model(image, meta_cat, meta_num, baseline_fvc_scaled, weeks_scaled)

        # Compute loss
        loss = laplace_log_likelihood_loss(mu, sigma, target_fvc_scaled)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update stats
        loss_meter.update(loss.item(), image.size(0))

    return loss_meter.avg


def evaluate(model, loader, scaler, device):
    """
    Evaluates the model on the validation set using the official metric.
    Predictions are unscaled back to original units (ml) before scoring.
    """
    model.eval()

    all_true = []
    all_pred = []
    all_sigma = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            meta_cat = batch["meta_cat"].to(device)
            meta_num = batch["meta_num"].to(device)
            baseline_fvc_scaled = batch["baseline_fvc_scaled"].to(device)
            weeks_scaled = batch["weeks_scaled"].to(device)

            # Forward pass
            mu_scaled, sigma_scaled = model(
                image, meta_cat, meta_num, baseline_fvc_scaled, weeks_scaled
            )

            # Move to CPU numpy
            mu_scaled = mu_scaled.cpu().numpy()
            sigma_scaled = sigma_scaled.cpu().numpy()
            raw_fvc = batch["raw_fvc"].numpy()

            # Unscale predictions
            mu_unscaled = scaler.unscale_fvc(mu_scaled)
            sigma_unscaled = scaler.unscale_sigma(sigma_scaled)

            all_true.extend(raw_fvc)
            all_pred.extend(mu_unscaled)
            all_sigma.extend(sigma_unscaled)

    # Compute official metric
    # Note: metric_laplace_log_likelihood handles clipping internally
    score = metric_laplace_log_likelihood(
        np.array(all_true), np.array(all_pred), np.array(all_sigma)
    )
    return score


def train():
    """
    Main training routine.
    """
    # 1. Setup
    Config.setup()
    device = Config.DEVICE
    print(f"Starting training on device: {device}")

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if Config.DEBUG:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Initialize Processors
    # CTScanProcessor handles image loading and caching
    processor = CTScanProcessor()

    # TabularScaler fits on training data to standardize inputs/targets
    scaler = TabularScaler()
    scaler.fit(train_df)

    # Baseline lookup for dataset construction
    # We combine lookups for both sets to ensure coverage
    train_lookup = get_baseline_lookup(train_df)
    val_lookup = get_baseline_lookup(val_df)

    # 4. Create Datasets and Loaders
    train_dataset = PulmonaryDataset(
        train_df, processor, scaler, train_lookup, mode="train"
    )
    val_dataset = PulmonaryDataset(val_df, processor, scaler, val_lookup, mode="val")

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

    # 5. Initialize Model
    model = CAPNet().to(device)

    # 6. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 7. Training Loop
    best_score = -float("inf")
    patience_counter = 0

    print(f"Training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = evaluate(model, val_loader, scaler, device)

        # Step scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New best model saved! Score: {best_score:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation Score: {best_score:.6f}")
