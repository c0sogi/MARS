import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import get_dataloaders
from library.model import MAZR_DS


class LaplaceNLLLoss(nn.Module):
    """
    Metric-Aligned Laplace Negative Log Likelihood Loss.
    Optimizes the objective: L = (sqrt(2) * |target - mu|) / sigma + log(sqrt(2) * sigma)
    """

    def __init__(self):
        super(LaplaceNLLLoss, self).__init__()

    def forward(self, mu, sigma, target):
        # Calculate absolute error
        delta = torch.abs(target - mu)

        # Constant sqrt(2)
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=mu.device))

        # Term 1: Scaled L1 Error
        term1 = (sqrt_2 * delta) / sigma

        # Term 2: Log Sigma Penalty
        term2 = torch.log(sqrt_2 * sigma)

        # Mean over batch
        loss = torch.mean(term1 + term2)
        return loss


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for imgs, tabular, targets in loader:
        imgs = imgs.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        mu, sigma = model(imgs, tabular)

        # Compute loss
        loss = criterion(mu, sigma, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device, stats):
    """
    Evaluates the model on the validation set.
    Computes both the NLL Loss and the Competition Metric (on raw scale).
    """
    model.eval()
    running_loss = 0.0

    # Containers for full-dataset metric calculation
    all_mu_raw = []
    all_sigma_raw = []
    all_targets_raw = []

    fvc_mean = stats["fvc_mean"]
    fvc_std = stats["fvc_std"]

    with torch.no_grad():
        for imgs, tabular, targets in loader:
            imgs = imgs.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)

            # Forward pass
            mu, sigma = model(imgs, tabular)

            # Compute Loss (on normalized scale)
            loss = criterion(mu, sigma, targets)
            running_loss += loss.item() * imgs.size(0)

            # Denormalize predictions for Metric Calculation
            # mu_raw = mu_norm * std + mean
            mu_raw = mu.cpu().numpy() * fvc_std + fvc_mean

            # sigma_raw = sigma_norm * std (scale only)
            sigma_raw = sigma.cpu().numpy() * fvc_std

            # target_raw = target_norm * std + mean
            target_raw = targets.cpu().numpy() * fvc_std + fvc_mean

            all_mu_raw.append(mu_raw)
            all_sigma_raw.append(sigma_raw)
            all_targets_raw.append(target_raw)

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_mu_raw = np.concatenate(all_mu_raw)
    all_sigma_raw = np.concatenate(all_sigma_raw)
    all_targets_raw = np.concatenate(all_targets_raw)

    # Calculate Competition Metric
    metric_score = calculate_metric(all_targets_raw, all_mu_raw, all_sigma_raw)

    return epoch_loss, metric_score


def run_training(load_cached_data=True):
    """
    Main execution function for training the MAZR-DS model.
    """
    seed_everything(Config.SEED)
    Config.setup()

    print(f"Initializing training on device: {Config.DEVICE}")

    # 1. Prepare Data
    train_loader, val_loader, stats = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=load_cached_data
    )

    # 2. Initialize Model
    device = torch.device(Config.DEVICE)
    model = MAZR_DS().to(device)

    # 3. Optimizer with Differential Learning Rates
    # Identify backbone parameters
    backbone_ids = list(map(id, model.backbone.parameters()))

    # Filter parameters into two groups
    backbone_params = model.backbone.parameters()
    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 4. Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 5. Loss Function
    criterion = LaplaceNLLLoss()

    # 6. Training Loop
    best_metric = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, device, stats)

        # Step Scheduler
        scheduler.step()

        # Checkpoint
        saved_msg = ""
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            saved_msg = " [Saved Best]"

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Metric: {val_metric:.10f} | "
            f"Time: {elapsed:.1f}s{saved_msg}"
        )

    print(f"Training complete. Best Validation Metric: {best_metric:.10f}")
    return best_metric
