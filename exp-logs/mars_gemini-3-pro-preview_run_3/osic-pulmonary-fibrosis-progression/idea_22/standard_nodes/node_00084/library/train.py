import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import SPPDSNet
from library.utils import calculate_metric


class MetricAlignedLaplaceLoss(nn.Module):
    """
    Loss function aligned with the competition metric:
    L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)

    This loss operates in the scaled space during training to ensure numerical stability.
    """

    def __init__(self):
        super().__init__()
        self.sqrt2 = torch.tensor(Config.SQRT2).to(Config.DEVICE)

    def forward(self, pred_mu, pred_sigma, target):
        """
        Args:
            pred_mu: Predicted FVC (scaled).
            pred_sigma: Predicted Confidence (scaled).
            target: True FVC (scaled).
        """
        # Calculate absolute error
        delta = torch.abs(target - pred_mu)

        # Term 1: Scaled error penalty
        term1 = (self.sqrt2 * delta) / pred_sigma

        # Term 2: Log uncertainty penalty
        term2 = torch.log(self.sqrt2 * pred_sigma)

        # Combine
        loss = torch.mean(term1 + term2)
        return loss


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        targets_scaled = batch["target_scaled"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        pred_mu, pred_sigma = model(images, tabular)

        # Ensure shapes match (flatten if necessary)
        pred_mu = pred_mu.squeeze()
        pred_sigma = pred_sigma.squeeze()
        targets_scaled = targets_scaled.squeeze()

        # Compute loss
        loss = criterion(pred_mu, pred_sigma, targets_scaled)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate_epoch(model, loader, preprocessor, device):
    """
    Runs validation, inverse transforms predictions, and calculates the competition metric.
    """
    model.eval()

    all_true_fvc = []
    all_pred_fvc = []
    all_pred_sigma = []

    # Get the scaler std for inverse transforming sigma
    # scale_ is an array, we take the first element as FVC is the first feature fitted in target_scaler
    fvc_scale = preprocessor.target_scaler.scale_[0]

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets_raw = batch["target"]  # Raw FVC values

            # Forward pass
            pred_mu_scaled, pred_sigma_scaled = model(images, tabular)

            # Move to CPU
            pred_mu_scaled = pred_mu_scaled.cpu().numpy().squeeze()
            pred_sigma_scaled = pred_sigma_scaled.cpu().numpy().squeeze()
            targets_raw = targets_raw.numpy().squeeze()

            # Inverse transform predictions
            # mu: standard inverse transform
            pred_mu_raw = preprocessor.inverse_transform_target(pred_mu_scaled)
            pred_mu_raw = pred_mu_raw.flatten()

            # sigma: multiply by standard deviation
            pred_sigma_raw = pred_sigma_scaled * fvc_scale

            all_true_fvc.extend(targets_raw)
            all_pred_fvc.extend(pred_mu_raw)
            all_pred_sigma.extend(pred_sigma_raw)

    # Convert to numpy arrays
    y_true = np.array(all_true_fvc)
    y_pred = np.array(all_pred_fvc)
    sigma_pred = np.array(all_pred_sigma)

    # Calculate metric
    score = calculate_metric(y_true, y_pred, sigma_pred)
    return score


def train_model(debug=False):
    """
    Main training loop.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    print(f"Starting training on device: {device}")
    print(f"Debug Mode: {debug}")

    # 2. Data
    train_loader, val_loader, test_loader, preprocessor = get_dataloaders(debug=debug)

    # 3. Model
    model = SPPDSNet().to(device)

    # 4. Optimization
    # Differential Learning Rates
    backbone_params = list(map(id, model.image_encoder.parameters()))
    head_params = filter(lambda p: id(p) not in backbone_params, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": model.image_encoder.parameters(), "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = MetricAlignedLaplaceLoss()

    # 5. Training Loop
    best_score = -float("inf")

    print("Epoch | Train Loss | Val Metric (LL)")
    print("-" * 35)

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate_epoch(model, val_loader, preprocessor, device)

        # Scheduler Step
        scheduler.step()

        # Checkpointing
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            saved_msg = " [Saved]"
        else:
            saved_msg = ""

        print(f"{epoch+1:03d}   | {train_loss:.6f}   | {val_score} {saved_msg}")

    print("-" * 35)
    print(f"Training Complete. Best Validation Score: {best_score}")

    return best_score
