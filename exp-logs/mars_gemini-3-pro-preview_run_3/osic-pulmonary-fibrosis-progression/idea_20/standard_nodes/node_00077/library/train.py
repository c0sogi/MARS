import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.model import TSCRNet
from library.data import get_dataloaders
from library.utils import (
    seed_everything,
    laplace_log_likelihood_metric,
    inverse_scale_predictions,
)


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Implements the Negative Laplace Log Likelihood Loss with the sqrt(2) correction.
    Loss = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()

    def forward(self, pred_mean, pred_sigma, target):
        """
        Args:
            pred_mean: Predicted mean (Z-score scaled)
            pred_sigma: Predicted std dev (Z-score scaled, positive)
            target: Ground truth (Z-score scaled)
        """
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=pred_mean.device))

        abs_diff = torch.abs(target - pred_mean)

        # Term 1: (sqrt(2) * |error|) / sigma
        term1 = (sqrt_2 * abs_diff) / pred_sigma

        # Term 2: ln(sqrt(2) * sigma)
        term2 = torch.log(sqrt_2 * pred_sigma)

        loss = term1 + term2
        return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Unpack batch
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        time_abs = batch["time_abs"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_mean, pred_sigma = model(images, tabular, time_abs)

        # Compute loss
        loss = criterion(pred_mean, pred_sigma, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluation loop. Computes Loss (on scaled data) and Metric (on original scale).
    """
    model.eval()
    running_loss = 0.0

    # Store predictions and targets for metric calculation
    all_pred_means = []
    all_pred_sigmas = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            time_abs = batch["time_abs"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            pred_mean, pred_sigma = model(images, tabular, time_abs)

            # Compute Loss (Standardized space)
            loss = criterion(pred_mean, pred_sigma, targets)
            running_loss += loss.item() * images.size(0)

            # Collect for Metric (Original space)
            all_pred_means.append(pred_mean.cpu())
            all_pred_sigmas.append(pred_sigma.cpu())
            all_targets.append(targets.cpu())

    # Aggregate
    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate tensors
    pred_means_t = torch.cat(all_pred_means)
    pred_sigmas_t = torch.cat(all_pred_sigmas)
    targets_t = torch.cat(all_targets)

    # Inverse Scale to Original Units (ml)
    pred_mean_orig, pred_sigma_orig = inverse_scale_predictions(
        pred_means_t, pred_sigmas_t
    )

    # Inverse scale targets: target_orig = target_scaled * std + mean
    targets_orig = targets_t * Config.TARGET_STD + Config.TARGET_MEAN

    # Compute Competition Metric
    metric_score = laplace_log_likelihood_metric(
        targets_orig, pred_mean_orig, pred_sigma_orig
    )

    return epoch_loss, metric_score


def run_training(debug=False):
    """
    Main orchestration function for training.
    """
    seed_everything(Config.SEED)

    # 1. Data
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # 2. Model
    model = TSCRNet()
    model.to(Config.DEVICE)

    # 3. Optimizer with Differential Learning Rates
    # Separate backbone parameters from the rest
    backbone_params = list(map(id, model.backbone.parameters()))
    head_params = filter(lambda p: id(p) not in backbone_params, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEADS},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 4. Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.N_EPOCHS, eta_min=1e-6
    )

    # 5. Loss
    criterion = LaplaceLogLikelihoodLoss()

    # 6. Training Loop
    best_metric = -float("inf")

    print(f"Starting training for {Config.N_EPOCHS} epochs on {Config.DEVICE}...")

    for epoch in range(Config.N_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )

        # Evaluate
        val_loss, val_metric = evaluate(model, val_loader, criterion, Config.DEVICE)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.N_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Metric: {val_metric:.6f}"
        )

        # Checkpoint
        if val_metric > best_metric:
            print(
                f"Metric improved ({best_metric:.6f} -> {val_metric:.6f}). Saving model..."
            )
            best_metric = val_metric
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )

    print(f"Training complete. Best Validation Metric: {best_metric:.6f}")
