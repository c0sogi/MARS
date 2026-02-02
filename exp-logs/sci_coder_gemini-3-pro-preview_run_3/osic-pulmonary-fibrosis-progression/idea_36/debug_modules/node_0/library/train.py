import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, AverageMeter, laplace_log_likelihood_metric
from library.data import get_dataloaders
from library.model import UCOSRNet


class LaplaceNLLLoss(nn.Module):
    """
    Metric-Aligned Laplace Log Likelihood Loss.
    L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)
    """

    def __init__(self, eps=1e-6):
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, target):
        mu, sigma = preds
        # sigma is strictly positive via Softplus in model, adding eps for numerical safety
        sigma = sigma + self.eps

        # Calculate absolute error
        delta = torch.abs(target - mu)

        # Calculate NLL
        # Note: We use the constants sqrt(2) as defined in the metric
        sqrt_2 = torch.tensor(np.sqrt(2.0), device=delta.device)

        nll = (sqrt_2 * delta) / sigma + torch.log(sqrt_2 * sigma)

        return torch.mean(nll)


def train_fn(dataloader, model, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()

    for batch in dataloader:
        imgs = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()

        mu, sigma = model(imgs, tabular)

        loss = criterion((mu, sigma), targets)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), imgs.size(0))

    return loss_meter.avg


def eval_fn(dataloader, model, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)
            targets_raw = batch["target_raw"].to(device)

            mu, sigma = model(imgs, tabular)

            # 1. Calculate Loss (in Scaled Space)
            loss = criterion((mu, sigma), targets)
            loss_meter.update(loss.item(), imgs.size(0))

            # 2. Calculate Metric (in Raw Space)
            # Inverse transform predictions
            mu_raw = mu * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_raw = sigma * Config.TARGET_STD

            # The metric function handles clipping and delta thresholding
            score = laplace_log_likelihood_metric(targets_raw, mu_raw, sigma_raw)
            metric_meter.update(score, imgs.size(0))

    return loss_meter.avg, metric_meter.avg


def run_training():
    seed_everything(Config.SEED)

    # Setup Directories
    checkpoint_dir = os.path.join(Config.WORKING_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load Data
    train_loader, val_loader = get_dataloaders(Config.TRAIN_CSV, Config.VAL_CSV)

    # Initialize Model
    device = torch.device(Config.DEVICE)
    model = UCOSRNet().to(device)

    # Optimizer with Differential Learning Rates
    # Filter parameters that require gradients (some backbone layers are frozen)
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Loss Function
    criterion = LaplaceNLLLoss()

    # Training Loop
    best_metric = -float("inf")
    patience = 10
    patience_counter = 0

    print(f"Starting training on device: {device}")
    print(f"Backbone LR: {Config.LR_BACKBONE}, Head LR: {Config.LR_HEAD}")

    for epoch in range(Config.EPOCHS):
        train_loss = train_fn(train_loader, model, criterion, optimizer, device)
        val_loss, val_metric = eval_fn(val_loader, model, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric: {val_metric}"
        )

        # Save Best Model (Higher Metric is Better)
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(
                model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth")
            )
            print(f"New best model saved! Metric: {best_metric}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Metric: {best_metric}")
