import os
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Import components from the provided library files
from library.utils import seed_everything, AverageMeter
from library.data import LungDataset, LungDataProcessor, get_transforms
from library.model import HighFidelityDualNet


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Custom loss function optimizing the modified Laplace Log Likelihood metric.

    The metric is defined as:
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Where:
        delta = min(|true - pred|, 1000)
        sigma_clipped = max(sigma, 70)

    The loss is the negative of the metric:
        loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super(LaplaceLogLikelihoodLoss, self).__init__()

    def forward(self, y_true, y_pred, sigma):
        # Ensure inputs are float and on the same device
        y_true = y_true.float()
        y_pred = y_pred.float()
        sigma = sigma.float()

        # Clipping sigma (confidence) at 70 ml
        sigma_clipped = torch.clamp(sigma, min=70)

        # Calculate absolute error
        delta = torch.abs(y_true - y_pred)

        # Clipping delta (error) at 1000 ml
        delta = torch.clamp(delta, max=1000)

        # Metric calculation components
        sqrt_2 = math.sqrt(2)

        # Loss = -Metric
        loss = (sqrt_2 * delta) / sigma_clipped + torch.log(sqrt_2 * sigma_clipped)

        return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """
    Executes one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        # Move data to device
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tab_vec = batch["tab_vec"].to(device)
        rel_week = batch["rel_week"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)
        baseline_fvc_sc = batch["baseline_fvc_sc"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        with autocast():
            # Forward pass
            pred_fvc, pred_conf = model(
                img_ax, img_cor, tab_vec, rel_week, baseline_fvc, baseline_fvc_sc
            )
            # Calculate loss
            loss = criterion(target, pred_fvc, pred_conf)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), img_ax.size(0))

    return loss_meter.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss (negative metric).
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tab_vec = batch["tab_vec"].to(device)
            rel_week = batch["rel_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            baseline_fvc_sc = batch["baseline_fvc_sc"].to(device)
            target = batch["target"].to(device)

            pred_fvc, pred_conf = model(
                img_ax, img_cor, tab_vec, rel_week, baseline_fvc, baseline_fvc_sc
            )

            loss = criterion(target, pred_fvc, pred_conf)
            loss_meter.update(loss.item(), img_ax.size(0))

    return loss_meter.avg


def run_training(
    train_path="./metadata/train.csv",
    val_path="./metadata/val.csv",
    cache_dir="./working/idea_14",
    epochs=20,
    batch_size=32,
    lr=1e-4,
    weight_decay=1e-5,
    patience=5,
    save_path="./working/best_model.pth",
):
    """
    Main driver function to setup data, model, and run the training loop with early stopping.
    """
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Metadata
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            "Metadata files not found. Please ensure they are generated."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # 2. Setup Data Processor and Datasets
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    processor = LungDataProcessor(cache_dir=cache_dir)
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    train_dataset = LungDataset(
        train_df, processor, transforms=train_transforms, mode="train"
    )
    val_dataset = LungDataset(val_df, processor, transforms=val_transforms, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 3. Setup Model, Optimizer, Scheduler, Loss
    model = HighFidelityDualNet().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = GradScaler()
    criterion = LaplaceLogLikelihoodLoss()

    # 4. Training Loop
    best_metric = -float("inf")
    patience_counter = 0

    print(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Convert loss back to metric for reporting (Metric = -Loss)
        val_metric = -val_loss

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Metric: {val_metric:.20f}"
        )

        # Early Stopping and Checkpointing
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"  -> Patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Metric: {best_metric:.20f}")
    return model
