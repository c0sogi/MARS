import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import seed_everything, AverageMeter, laplace_log_likelihood_metric
from library.data import LungDataset, get_transforms
from library.model import GTVRNet


class LaplaceLikelihoodLoss(nn.Module):
    """
    Custom Loss function optimizing the modified Laplace Log Likelihood.
    Loss = -Metric
    """

    def __init__(self):
        super().__init__()
        self.sqrt2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, fvc_pred, sigma_pred, fvc_true):
        # Ensure device consistency
        device = fvc_pred.device
        sqrt2 = self.sqrt2.to(device)

        # Clip sigma (confidence)
        sigma_clipped = torch.clamp(sigma_pred, min=Config.MIN_CONFIDENCE)

        # Calculate absolute error
        abs_error = torch.abs(fvc_true - fvc_pred)

        # Clip error (delta)
        delta = torch.clamp(abs_error, max=Config.MAX_ERROR)

        # Calculate Negative Log Likelihood
        # Metric formula: - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        # Loss = -Metric = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)
        term1 = (sqrt2 * delta) / sigma_clipped
        term2 = torch.log(sqrt2 * sigma_clipped)

        loss = term1 + term2
        return torch.mean(loss)


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        # Move data to device
        img_axial = batch["img_axial"].to(device)
        img_coronal = batch["img_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        meta = batch["meta"].to(device)
        target = batch["target"].to(device)  # Shape (B, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        fvc_pred, sigma_pred = model(img_axial, img_coronal, tabular, meta)

        # Reshape predictions to match target (B, 1)
        fvc_pred = fvc_pred.view(-1, 1)
        sigma_pred = sigma_pred.view(-1, 1)

        # Compute loss
        loss = criterion(fvc_pred, sigma_pred, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update stats
        loss_meter.update(loss.item(), img_axial.size(0))

    return loss_meter.avg


def valid_epoch(model, loader, device):
    """
    Performs one epoch of validation.
    Returns the average metric score.
    """
    model.eval()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            fvc_pred, sigma_pred = model(img_axial, img_coronal, tabular, meta)

            fvc_pred = fvc_pred.view(-1, 1)
            sigma_pred = sigma_pred.view(-1, 1)

            # Calculate metric
            # Using the utility function provided in library
            score = laplace_log_likelihood_metric(target, fvc_pred, sigma_pred)

            metric_meter.update(score, img_axial.size(0))

    return metric_meter.avg


def run_training():
    """
    Main orchestration function for training the GTVR-Net.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    Config.make_dirs()
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")

    # 2. Data Preparation
    train_dataset = LungDataset(
        csv_path=Config.TRAIN_CSV,
        mode="train",
        transform=get_transforms("train"),
        load_cached_data=True,
    )

    val_dataset = LungDataset(
        csv_path=Config.VAL_CSV,
        mode="val",
        transform=get_transforms("val"),
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Model, Optimizer, Loss
    model = GTVRNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLikelihoodLoss()

    # 4. Training Loop with Early Stopping
    best_metric = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.MODEL_SAVE_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metric = valid_epoch(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Metric: {val_metric:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_metric > best_metric:
            print(
                f"Validation metric improved ({best_metric:.6f} --> {val_metric:.6f}). Saving model..."
            )
            best_metric = val_metric
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation Metric: {best_metric:.6f}")
    print(f"Best model saved to: {best_model_path}")
