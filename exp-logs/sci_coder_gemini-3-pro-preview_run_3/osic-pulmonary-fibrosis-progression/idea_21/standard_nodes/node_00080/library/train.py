import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, metric_laplace_log_likelihood
from library.data import get_dataloaders
from library.model import RCRFNet


class LaplaceNLLLoss(nn.Module):
    """
    Metric-Aligned Laplace Negative Log Likelihood Loss.

    Formula: L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)

    This aligns with the competition metric which assumes a Laplace distribution.
    We minimize this loss during training.
    """

    def __init__(self):
        super(LaplaceNLLLoss, self).__init__()

    def forward(self, mu, sigma, target):
        """
        Args:
            mu: Predicted mean (B, 1)
            sigma: Predicted standard deviation (B, 1)
            target: Ground truth (B, 1)
        """
        # Ensure shapes match
        if target.dim() == 1:
            target = target.view(-1, 1)

        # Calculate absolute error
        delta = torch.abs(target - mu)

        # Constant sqrt(2)
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=mu.device))

        # Loss calculation
        # Term 1: (sqrt(2) * delta) / sigma
        term1 = (sqrt_2 * delta) / sigma

        # Term 2: ln(sqrt(2) * sigma)
        term2 = torch.log(sqrt_2 * sigma)

        loss = term1 + term2

        return torch.mean(loss)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (imgs, clinical, t_rel, targets) in enumerate(loader):
        imgs = imgs.to(device)
        clinical = clinical.to(device)
        t_rel = t_rel.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()

        # Forward pass
        mu, sigma = model(imgs, clinical, t_rel)

        # Compute loss
        loss = criterion(mu, sigma, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Performs inverse transformation to calculate the true metric in ml.
    """
    model.eval()

    all_mu = []
    all_sigma = []
    all_targets = []

    with torch.no_grad():
        for imgs, clinical, t_rel, targets in loader:
            imgs = imgs.to(device)
            clinical = clinical.to(device)
            t_rel = t_rel.to(device)

            # Forward pass (outputs are scaled)
            mu_scaled, sigma_scaled = model(imgs, clinical, t_rel)

            # Collect results
            all_mu.append(mu_scaled.cpu().numpy())
            all_sigma.append(sigma_scaled.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate
    mu_scaled = np.concatenate(all_mu).flatten()
    sigma_scaled = np.concatenate(all_sigma).flatten()
    targets_scaled = np.concatenate(all_targets).flatten()

    # -------------------------------------------------------------------------
    # Inverse Transformation
    # -------------------------------------------------------------------------
    # mu_final = mu_scaled * std + mean
    mu_final = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN

    # sigma_final = sigma_scaled * std
    sigma_final = sigma_scaled * Config.TARGET_STD

    # target_final = target_scaled * std + mean
    targets_final = targets_scaled * Config.TARGET_STD + Config.TARGET_MEAN

    # -------------------------------------------------------------------------
    # Metric Calculation
    # -------------------------------------------------------------------------
    score = metric_laplace_log_likelihood(targets_final, mu_final, sigma_final)

    return score


def run_training():
    """
    Main orchestration function for training.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model
    print("Initializing RCRF-Net...")
    model = RCRFNet().to(device)

    # 4. Optimizer with Differential Learning Rates
    # Separate backbone parameters from the rest
    backbone_params = list(model.backbone.parameters())
    backbone_ids = list(map(id, backbone_params))

    head_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEADS},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 5. Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 6. Loss
    criterion = LaplaceNLLLoss()

    # 7. Training Loop
    best_score = -float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        end_time = time.time()
        epoch_mins = (end_time - start_time) / 60

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Score: {val_score} | "  # Full precision as requested
            f"Time: {epoch_mins:.2f} min"
        )

        # Early Stopping & Checkpointing
        if val_score > best_score:
            print(f"Score improved from {best_score} to {val_score}. Saving model...")
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
