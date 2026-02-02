import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    save_checkpoint,
    laplace_log_likelihood_metric,
)
from library.data import get_dataloaders
from library.model import PRTNet


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Computes the Laplace Log Likelihood Loss.
    Loss = (sqrt(2) * |y - mu|) / sigma + ln(sqrt(2) * sigma)

    This is used for training on standardized targets.
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, mu, sigma, target):
        # Ensure sqrt_2 is on the correct device
        if self.sqrt_2.device != mu.device:
            self.sqrt_2 = self.sqrt_2.to(mu.device)

        # Calculate absolute error
        delta = torch.abs(target - mu)

        # Calculate NLL terms
        # Term 1: (sqrt(2) * delta) / sigma
        term1 = (self.sqrt_2 * delta) / sigma

        # Term 2: ln(sqrt(2) * sigma)
        term2 = torch.log(self.sqrt_2 * sigma)

        loss = term1 + term2
        return torch.mean(loss)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Handles the training of one epoch.
    """
    model.train()
    losses = AverageMeter()

    # Iterate over batches
    # Using tqdm for visual feedback is allowed, but we'll keep it minimal/silent if needed
    # The prompt asks not to print progress bars, so we iterate directly.
    for batch_idx, batch in enumerate(loader):
        # Move data to device
        images = batch["image"].to(device)
        static = batch["static"].to(device)
        rel_time = batch["rel_time"].to(device)
        targets = batch["target"].to(device)

        # Forward pass
        mu, sigma = model(images, static, rel_time)

        # Compute loss
        loss = criterion(mu, sigma, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_one_epoch(model, loader, device, fvc_stats):
    """
    Handles validation.
    Performs inverse transformation of predictions to calculate the official metric.
    """
    model.eval()
    metric_score = AverageMeter()

    # Unpack scaling stats
    fvc_mean = fvc_stats["fvc_mean"]
    fvc_std = fvc_stats["fvc_std"]

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            static = batch["static"].to(device)
            rel_time = batch["rel_time"].to(device)
            # We need raw_fvc for the metric calculation, not the scaled target
            raw_fvc = batch["raw_fvc"].to(device)

            # Forward pass (outputs are standardized)
            mu_scaled, sigma_scaled = model(images, static, rel_time)

            # Inverse Transform to Absolute Scale (ml)
            # mu_abs = mu_scaled * std + mean
            mu_abs = mu_scaled * fvc_std + fvc_mean

            # sigma_abs = sigma_scaled * std
            # Note: sigma scales linearly with the standard deviation
            sigma_abs = sigma_scaled * fvc_std

            # Calculate Official Metric
            # The metric function handles clipping internally
            score = laplace_log_likelihood_metric(raw_fvc, mu_abs, sigma_abs)

            metric_score.update(score, images.size(0))

    return metric_score.avg


def run_training():
    """
    Main driver function for the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training for {Config.PROJECT_NAME}")
    print(f"Device: {device}")

    # 2. Data Loading
    # load_cached_data=True allows using pre-processed .npy files if they exist
    train_loader, val_loader, _, stats = get_dataloaders(load_cached_data=True)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # 3. Model Initialization
    model = PRTNet().to(device)

    # 4. Optimizer & Scheduler
    # Differential Learning Rates
    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": Config.LR_BACKBONE},
            {"params": model.img_projector.parameters(), "lr": Config.LR_HEAD},
            {"params": model.mlp.parameters(), "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    criterion = LaplaceLogLikelihoodLoss()

    # 5. Training Loop
    best_score = -float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_score = validate_one_epoch(model, val_loader, device, stats)

        # Step Scheduler
        scheduler.step()

        # Checkpoint
        is_best = val_score > best_score
        if is_best:
            best_score = val_score

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_score": best_score,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
        )

        # Logging
        # Print full precision as requested
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Score: {val_score} | "
            f"Best Score: {best_score}"
        )

    print("Training complete.")
    print(f"Best Validation Score: {best_score}")
