import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.data import OSICDataset, get_transforms
from library.model import LARFNet


class LaplaceLoss(nn.Module):
    """
    Implements the negative of the modified Laplace Log Likelihood metric.
    Minimizing this loss is equivalent to maximizing the competition metric.

    Formula:
        Loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    Where:
        delta = min(|true - pred|, 1000)
        sigma_clipped = max(sigma, 70)
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, y_sigma):
        # Ensure inputs are on the correct device and shape
        # y_true, y_pred, y_sigma shape: (Batch_Size,)

        # 1. Clip confidence values (sigma) to a minimum of 70
        sigma_clipped = torch.clamp(y_sigma, min=Config.MIN_CONFIDENCE_CLIP)

        # 2. Calculate absolute error (delta) and clip it to a maximum of 1000
        delta = torch.abs(y_true - y_pred)
        delta = torch.clamp(delta, max=Config.MAX_ERROR_CLIP)

        # 3. Calculate the negative metric (Loss)
        sqrt_2 = torch.sqrt(torch.tensor(2.0, device=y_true.device))

        term1 = (sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(sqrt_2 * sigma_clipped)

        loss = term1 + term2

        # Return the mean loss over the batch
        return torch.mean(loss)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        img_axial = batch["image_axial"].to(device)
        img_coronal = batch["image_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        week = batch["week"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # We pass week and baseline_fvc to get direct predictions (fvc, sigma)
        fvc_pred, sigma_pred = model(
            img_axial, img_coronal, tabular, week=week, baseline_fvc=baseline_fvc
        )

        # Calculate loss
        loss = criterion(target, fvc_pred, sigma_pred)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * target.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Performs validation and calculates the official metric.
    """
    model.eval()
    running_loss = 0.0

    # Lists to store predictions for metric calculation
    all_true = []
    all_pred = []
    all_sigma = []

    with torch.no_grad():
        for batch in loader:
            img_axial = batch["image_axial"].to(device)
            img_coronal = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            week = batch["week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            # Forward pass
            fvc_pred, sigma_pred = model(
                img_axial, img_coronal, tabular, week=week, baseline_fvc=baseline_fvc
            )

            # Calculate loss
            loss = criterion(target, fvc_pred, sigma_pred)
            running_loss += loss.item() * target.size(0)

            # Store for metric calculation
            all_true.append(target.cpu().numpy())
            all_pred.append(fvc_pred.cpu().numpy())
            all_sigma.append(sigma_pred.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    all_sigma = np.concatenate(all_sigma)

    # Calculate official metric
    metric_score = calculate_metric(all_true, all_pred, all_sigma)

    return epoch_loss, metric_score


def run_training():
    """
    Main function to execute the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Ensure working directory exists for checkpoints
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    checkpoint_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training on device: {device}")
    Config.print_config()

    # 2. Prepare Data
    # Transforms
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")

    # Datasets
    train_dataset = OSICDataset(
        csv_path=Config.TRAIN_CSV, mode="train", transform=train_transform
    )
    val_dataset = OSICDataset(
        csv_path=Config.VAL_CSV, mode="val", transform=val_transform
    )

    # Debugging subset logic
    if Config.DEBUG:
        print(f"DEBUG MODE: Truncating datasets to {Config.DEBUG_SUBSET_SIZE} samples.")
        indices = list(range(min(len(train_dataset), Config.DEBUG_SUBSET_SIZE)))
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        val_indices = list(range(min(len(val_dataset), Config.DEBUG_SUBSET_SIZE)))
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 3. Initialize Model, Loss, Optimizer
    model = LARFNet().to(device)
    criterion = LaplaceLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_metric = -float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metric = validate_one_epoch(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric: {val_metric:.10f}"
        )

        # 5. Early Stopping & Checkpointing
        # Metric is negative, higher is better (e.g., -6.5 is better than -6.8)
        if val_metric > best_metric:
            print(
                f"  -> Metric Improved ({best_metric:.6f} -> {val_metric:.6f}). Saving model..."
            )
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("\nTraining complete.")
    print(f"Best Validation Metric: {best_metric:.10f}")
    print(f"Best model saved to: {checkpoint_path}")
