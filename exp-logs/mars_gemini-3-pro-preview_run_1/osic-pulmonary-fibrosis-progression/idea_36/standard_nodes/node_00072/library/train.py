import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    LaplaceLogLikelihoodLoss,
    compute_metric_score,
)
from library.data import get_dataloaders
from library.model import SCVRNet


def train_one_epoch(epoch, model, loader, optimizer, criterion, device):
    """
    Handles the training loop for a single epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, data in enumerate(loader):
        # Move data to device
        img_ax = data["img_ax"].to(device)
        img_cor = data["img_cor"].to(device)
        tabular = data["tabular"].to(device)

        target_fvc = data["target_fvc"].to(device)
        week_delta = data["week_delta"].to(device)
        baseline_fvc = data["baseline_fvc"].to(device)

        optimizer.zero_grad()

        # Forward pass: Get parameters (alpha, sigma_base, sigma_growth)
        # Output shape: (Batch, 3)
        params = model(img_ax, img_cor, tabular)

        alpha = params[:, 0]
        sigma_base = params[:, 1]
        sigma_growth = params[:, 2]

        # Reconstruct predictions based on linear trajectory assumption
        # FVC_pred = Baseline + alpha * delta_t
        fvc_pred = baseline_fvc + alpha * week_delta

        # Sigma_pred = Sigma_base + Sigma_growth * |delta_t|
        sigma_pred = sigma_base + sigma_growth * torch.abs(week_delta)

        # Compute Loss
        loss = criterion(fvc_pred, target_fvc, sigma_pred)

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), img_ax.size(0))

    return loss_meter.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for data in loader:
            img_ax = data["img_ax"].to(device)
            img_cor = data["img_cor"].to(device)
            tabular = data["tabular"].to(device)

            target_fvc = data["target_fvc"].to(device)
            week_delta = data["week_delta"].to(device)
            baseline_fvc = data["baseline_fvc"].to(device)

            # Forward pass
            params = model(img_ax, img_cor, tabular)

            alpha = params[:, 0]
            sigma_base = params[:, 1]
            sigma_growth = params[:, 2]

            # Reconstruct predictions
            fvc_pred = baseline_fvc + alpha * week_delta
            sigma_pred = sigma_base + sigma_growth * torch.abs(week_delta)

            # Compute Metric Score
            # Note: compute_metric_score handles the clipping internally
            score = compute_metric_score(fvc_pred, target_fvc, sigma_pred)

            metric_meter.update(score, img_ax.size(0))

    return metric_meter.avg


def run_training(save_path="./working/best_model.pth"):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")

    # 2. Data
    train_loader, val_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model
    model = SCVRNet()
    model = model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # Loss function for training (allows gradients on errors > 1000)
    criterion = LaplaceLogLikelihoodLoss(for_training=True)

    # 5. Training Loop
    best_metric = -float("inf")
    patience_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, device
        )

        # Validate
        val_metric = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Metric: {val_metric:.10f}"
        )

        # Checkpoint & Early Stopping
        if val_metric > best_metric:
            print(
                f"New best metric! ({best_metric:.6f} -> {val_metric:.6f}). Saving model..."
            )
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Metric: {best_metric:.10f}")
    return best_metric
