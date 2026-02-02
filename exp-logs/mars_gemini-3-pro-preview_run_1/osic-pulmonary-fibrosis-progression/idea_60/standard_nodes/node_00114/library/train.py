import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    AverageMeter,
    seed_everything,
    laplace_log_likelihood,
    save_checkpoint,
    log_message,
)
from library.data import LungDataset, get_transforms
from library.model import DCSLNet


class LaplaceLoss(nn.Module):
    """
    Differentiable implementation of the Negative Modified Laplace Log Likelihood.
    Loss = -Metric, since we want to maximize the Metric.

    Metric formula:
        sigma_clipped = max(sigma, 70)
        delta = min(|true - pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLoss, self).__init__()
        self.sqrt_2 = np.sqrt(2)

    def forward(self, fvc_true, fvc_pred, sigma):
        # 1. Clip confidence (sigma) at 70 ml
        # Use clamp for differentiability (gradients zero out below 70, which is desired)
        sigma_clipped = torch.clamp(sigma, min=Config.MIN_CONFIDENCE_CLIP)

        # 2. Calculate absolute error (delta) and clip at 1000 ml
        abs_error = torch.abs(fvc_true - fvc_pred)
        delta = torch.clamp(abs_error, max=Config.MAX_ERROR_CLIP)

        # 3. Compute Negative Metric (Loss)
        # Loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
        term1 = (self.sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        loss = term1 + term2
        return torch.mean(loss)


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Handles the training loop for a single epoch.
    """
    model.train()
    losses = AverageMeter("Loss", ":.4f")

    for i, batch in enumerate(train_loader):
        # Move inputs to device
        img_axial = batch["img_axial"].to(device)
        img_coronal = batch["img_coronal"].to(device)
        tabular = batch["tabular"].to(device)

        # Move metadata/targets to device
        meta_dt = batch["meta_dt"].to(device)  # Delta time (weeks)
        meta_base_fvc = batch["meta_base_fvc"].to(device)  # Baseline FVC
        target_fvc = batch["target"].to(device)  # True FVC at current week

        # Forward Pass
        # Output: [alpha, sigma_base, sigma_growth]
        preds = model(img_axial, img_coronal, tabular)

        alpha = preds[:, 0]
        sigma_base = preds[:, 1]
        sigma_growth = preds[:, 2]

        # Reconstruct Trajectory Predictions
        # FVC_pred = Baseline + alpha * dt
        fvc_pred = meta_base_fvc + alpha * meta_dt

        # Sigma_pred = Base + Growth * |dt|
        sigma_pred = sigma_base + sigma_growth * torch.abs(meta_dt)

        # Compute Loss
        loss = criterion(target_fvc, fvc_pred, sigma_pred)

        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img_axial.size(0))

    return losses.avg


def validate(val_loader, model, device):
    """
    Evaluates the model on the validation set using the official metric.
    """
    model.eval()

    # Store all predictions and targets to compute metric globally (or batch-wise average)
    # The metric is defined as the average across all Patient_Weeks.
    all_fvc_true = []
    all_fvc_pred = []
    all_sigma_pred = []

    with torch.no_grad():
        for batch in val_loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)

            meta_dt = batch["meta_dt"].to(device)
            meta_base_fvc = batch["meta_base_fvc"].to(device)
            target_fvc = batch["target"].to(device)

            # Forward Pass
            preds = model(img_axial, img_coronal, tabular)

            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            # Reconstruct
            fvc_pred = meta_base_fvc + alpha * meta_dt
            sigma_pred = sigma_base + sigma_growth * torch.abs(meta_dt)

            # Collect
            all_fvc_true.append(target_fvc.cpu().numpy())
            all_fvc_pred.append(fvc_pred.cpu().numpy())
            all_sigma_pred.append(sigma_pred.cpu().numpy())

    # Concatenate
    all_fvc_true = np.concatenate(all_fvc_true)
    all_fvc_pred = np.concatenate(all_fvc_pred)
    all_sigma_pred = np.concatenate(all_sigma_pred)

    # Compute Metric
    score = laplace_log_likelihood(all_fvc_true, all_fvc_pred, all_sigma_pred)

    return score


def train_model(debug=False, epochs=Config.EPOCHS):
    """
    Main function to train the DCSL-Net model.
    """
    seed_everything(Config.SEED)
    log_message(f"Starting training for experiment: {Config.EXPERIMENT_NAME}")
    log_message(f"Device: {Config.DEVICE}")

    # 1. Prepare Datasets
    train_dataset = LungDataset(
        Config.TRAIN_CSV, mode="train", transform=get_transforms("train")
    )
    val_dataset = LungDataset(
        Config.VAL_CSV, mode="val", transform=get_transforms("val")
    )

    # Debug Mode: Truncate datasets
    if debug:
        log_message("DEBUG MODE: Truncating datasets to 50 samples.")
        train_dataset.df = train_dataset.df.iloc[:50]
        val_dataset.df = val_dataset.df.iloc[:50]

    # 2. DataLoaders
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

    # 3. Model, Criterion, Optimizer
    model = DCSLNet().to(Config.DEVICE)
    criterion = LaplaceLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_score = -float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, Config.DEVICE, epoch
        )

        # Validate
        val_score = validate(val_loader, model, Config.DEVICE)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        log_message(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Score: {val_score:.10f}"  # Full precision as requested
        )

        # Checkpointing & Early Stopping
        is_best = val_score > best_score
        if is_best:
            best_score = val_score
            patience_counter = 0
            log_message(f"New best model found! Score: {best_score:.10f}")
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_score": best_score,
            },
            is_best,
            filename=f"epoch_{epoch+1}.pth",
        )

        if patience_counter >= Config.PATIENCE:
            log_message(f"Early stopping triggered after {epoch+1} epochs.")
            break

    log_message(f"Training complete. Best Validation Score: {best_score:.10f}")
    return best_score
