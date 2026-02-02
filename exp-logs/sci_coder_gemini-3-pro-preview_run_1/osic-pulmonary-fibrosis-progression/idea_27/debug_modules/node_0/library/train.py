import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import VCDAN
from library.utils import AverageMeter, calculate_metric


class LaplaceLoss(nn.Module):
    """
    Differentiable implementation of the Negative Modified Laplace Log Likelihood.
    Optimizing this minimizes the negative metric, effectively maximizing the score.
    """

    def __init__(self):
        super(LaplaceLoss, self).__init__()

    def forward(self, pred_fvc, pred_conf, target_fvc):
        # Clipping logic matches the metric definition
        # sigma_clipped = max(sigma, 70)
        sigma_clipped = torch.clamp(pred_conf, min=Config.MIN_CONFIDENCE)

        # delta = min(|true - pred|, 1000)
        delta = torch.abs(target_fvc - pred_fvc)
        delta_clipped = torch.clamp(delta, max=Config.MAX_ERROR)

        sqrt_2 = math.sqrt(2)

        # Metric formula: - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        metric = -(sqrt_2 * delta_clipped) / sigma_clipped - torch.log(
            sqrt_2 * sigma_clipped
        )

        # Return negative mean metric to minimize loss
        return -torch.mean(metric)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Handles one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        # Move data to device
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        fvc_pred, conf_pred = model(img_ax, img_cor, tabular)

        # Compute loss
        loss = criterion(fvc_pred, conf_pred, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update stats
        loss_meter.update(loss.item(), img_ax.size(0))

    return loss_meter.avg


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set using the official metric.
    """
    model.eval()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            fvc_pred, conf_pred = model(img_ax, img_cor, tabular)

            # Calculate metric using the utility function (handles numpy conversion)
            score = calculate_metric(target, fvc_pred, conf_pred)

            metric_meter.update(score, img_ax.size(0))

    return metric_meter.avg


def run_training():
    """
    Main training routine with Early Stopping and Checkpointing.
    """
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Create working directories
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    print("Loading Data...")
    # Load dataloaders (caching handled internally by library.data)
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    print("Initializing Model...")
    device = torch.device(Config.DEVICE)
    model = VCDAN().to(device)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    criterion = LaplaceLoss()

    # Tracking variables
    best_metric = -float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs.")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metric = evaluate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Metric: {val_metric}"
        )

        # Checkpointing and Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved! Metric: {best_metric}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    print(f"Training Complete. Best Validation Metric: {best_metric}")
