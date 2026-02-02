import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import (
    DEVICE,
    EPOCHS,
    LR_BACKBONE,
    LR_HEAD,
    WEIGHT_DECAY,
    T_MAX,
    ETA_MIN,
    CHECKPOINT_DIR,
    TARGET_MEAN,
    TARGET_STD,
    DEBUG,
)
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.model import RSTCNet


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Loss function for Laplace Distribution.
    Minimizes: (sqrt(2) * |y - mu|) / sigma + log(sqrt(2) * sigma)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()

    def forward(self, mu, sigma, target):
        # delta = |y - mu|
        delta = torch.abs(target - mu)

        # sigma is already positive (softplus + eps) from the model output

        # Move constant to device
        sqrt_2 = torch.tensor(np.sqrt(2), device=delta.device)

        # Term 1: Error scaled by uncertainty
        term1 = (sqrt_2 * delta) / sigma

        # Term 2: Log uncertainty penalty
        term2 = torch.log(sqrt_2 * sigma)

        # Mean over batch
        loss = torch.mean(term1 + term2)
        return loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for batch in loader:
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        time = batch["time"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        mu, sigma = model(images, tabular, time)

        loss = criterion(mu, sigma, targets)
        loss.backward()

        optimizer.step()

        # Accumulate batch loss scaled by batch size
        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Performs validation and calculates the competition metric.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    all_preds_mu = []
    all_preds_sigma = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            time = batch["time"].to(device)
            targets = batch["target"].to(device)

            mu, sigma = model(images, tabular, time)

            loss = criterion(mu, sigma, targets)
            running_loss += loss.item() * images.size(0)

            # Collect predictions for metric calculation
            all_preds_mu.append(mu.cpu().numpy())
            all_preds_sigma.append(sigma.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds_mu = np.concatenate(all_preds_mu)
    all_preds_sigma = np.concatenate(all_preds_sigma)
    all_targets = np.concatenate(all_targets)

    # --- Inverse Transformation ---
    # The model predicts Z-scored FVC. We must convert back to mL for the metric.
    # y_orig = y_norm * std + mean
    # sigma_orig = sigma_norm * std (Scaling only, no shift)

    preds_mu_orig = all_preds_mu * TARGET_STD + TARGET_MEAN
    preds_sigma_orig = all_preds_sigma * TARGET_STD
    targets_orig = all_targets * TARGET_STD + TARGET_MEAN

    # Calculate competition metric
    # Metric expects flattened arrays
    metric_score = calculate_metric(
        targets_orig.flatten(), preds_mu_orig.flatten(), preds_sigma_orig.flatten()
    )

    epoch_loss = running_loss / dataset_size
    return epoch_loss, metric_score


def run_training():
    """
    Main orchestration function for training.
    """
    seed_everything()

    # 1. Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True, debug=DEBUG)

    # 2. Initialize Model
    model = RSTCNet().to(DEVICE)

    # 3. Optimizer with Differential Learning Rates
    # Separate backbone parameters (frozen/unfrozen parts) from the new head
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

    # 4. Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_MAX, eta_min=ETA_MIN
    )

    # 5. Loss Function
    criterion = LaplaceLogLikelihoodLoss()

    # 6. Training Loop
    best_metric = -float("inf")
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {DEVICE} for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_metric = validate_one_epoch(model, val_loader, criterion, DEVICE)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Metric: {val_metric}"
        )

        # Save best model (Metric is negative, higher is better)
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with metric: {best_metric}")

    print(f"Training complete. Best Metric: {best_metric}")
