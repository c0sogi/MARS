import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.model import MAOPDSNet


class MetricAlignedLLLoss(nn.Module):
    """
    Metric-Aligned Laplace Log Likelihood Loss.
    Formula: L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self):
        super(MetricAlignedLLLoss, self).__init__()

    def forward(self, outputs, targets):
        """
        Args:
            outputs (torch.Tensor): Shape (B, 2). Column 0 is mu, Column 1 is raw_sigma.
            targets (torch.Tensor): Shape (B,). Normalized target values.
        """
        mu = outputs[:, 0]
        raw_sigma = outputs[:, 1]

        # Enforce positivity for sigma using Softplus
        # Adding a small epsilon for numerical stability
        sigma = F.softplus(raw_sigma) + 1e-6

        # Calculate absolute error
        abs_error = torch.abs(targets - mu)

        # Constants
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=outputs.device))

        # Loss terms
        term1 = (sqrt_2 * abs_error) / sigma
        term2 = torch.log(sqrt_2 * sigma)

        loss = term1 + term2

        return torch.mean(loss)


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        outputs = model(images, tabular)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def validate_epoch(model, loader, criterion, device):
    """
    Validates the model and calculates the competition metric.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # Lists to store raw values for metric calculation
    all_targets_raw = []
    all_mu_raw = []
    all_sigma_raw = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)
            batch_size = images.size(0)

            outputs = model(images, tabular)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Inverse transform for metric calculation
            mu_scaled = outputs[:, 0].cpu().numpy()
            raw_sigma_scaled = outputs[:, 1]
            sigma_scaled = F.softplus(raw_sigma_scaled).cpu().numpy() + 1e-6

            targets_scaled = targets.cpu().numpy()

            # Unscale using Config constants
            mu_raw = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_raw = sigma_scaled * Config.TARGET_STD
            targets_raw = targets_scaled * Config.TARGET_STD + Config.TARGET_MEAN

            all_mu_raw.extend(mu_raw)
            all_sigma_raw.extend(sigma_raw)
            all_targets_raw.extend(targets_raw)

    avg_loss = running_loss / dataset_size

    # Calculate competition metric
    metric_score = calculate_metric(
        np.array(all_targets_raw), np.array(all_mu_raw), np.array(all_sigma_raw)
    )

    return avg_loss, metric_score


def run_training(patience=10):
    """
    Orchestrates the training process.
    """
    seed_everything(Config.SEED)

    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader, stats = get_dataloaders()

    # Initialize Model
    device = torch.device(Config.DEVICE)
    print(f"Initializing model on {device}...")
    model = MAOPDSNet().to(device)

    # Optimizer with differential learning rates
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
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # Loss
    criterion = MetricAlignedLLLoss()

    best_metric = -float("inf")
    epochs_no_improve = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metric = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Metric: {val_metric:.10f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  >>> New Best Model Saved! Metric: {best_metric:.10f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Metric: {best_metric:.10f}")
