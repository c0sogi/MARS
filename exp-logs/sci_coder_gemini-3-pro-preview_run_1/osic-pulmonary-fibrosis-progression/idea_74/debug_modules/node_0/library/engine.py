import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import (
    seed_everything,
    LaplaceLogLikelihoodLoss,
    compute_metric_score,
)
from library.data import get_dataloaders
from library.model import PCCGNet


def train_one_epoch(model, loader, optimizer, device, loss_fn):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tabular = batch["tabular"].to(device)
        weeks = batch["weeks"].to(device)
        base_fvc = batch["base_fvc"].to(device)
        targets = batch["target"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        # Model expects: img_ax, img_cor, tabular, weeks, base_fvc
        preds = model(img_ax, img_cor, tabular, weeks, base_fvc)

        # Compute loss
        loss = loss_fn(preds, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img_ax.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, device, loss_fn):
    """
    Evaluates the model on the validation set.
    Returns average loss and average competition metric score.
    """
    model.eval()
    running_loss = 0.0
    running_metric = 0.0

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            preds = model(img_ax, img_cor, tabular, weeks, base_fvc)

            # Compute Loss
            loss = loss_fn(preds, targets)
            running_loss += loss.item() * img_ax.size(0)

            # Compute Metric Score
            metric = compute_metric_score(preds, targets)
            running_metric += metric * img_ax.size(0)

    avg_loss = running_loss / len(loader.dataset)
    avg_metric = running_metric / len(loader.dataset)

    return avg_loss, avg_metric


def predict(model, loader, device):
    """
    Generates predictions for a given loader.
    Returns arrays of FVC and Confidence.
    """
    model.eval()
    fvc_preds = []
    conf_preds = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)

            preds = model(img_ax, img_cor, tabular, weeks, base_fvc)

            fvc_preds.append(preds[:, 0].cpu().numpy())
            conf_preds.append(preds[:, 1].cpu().numpy())

    return np.concatenate(fvc_preds), np.concatenate(conf_preds)


def run_training(debug=False):
    """
    Main driver for training the PCCG-Net model.
    """
    # 1. Setup
    Config.setup()
    device = torch.device(Config.DEVICE)
    seed_everything(Config.SEED)

    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    print("Initializing Model...")
    model = PCCGNet()
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Loss Function
    loss_fn = LaplaceLogLikelihoodLoss()
    loss_fn.to(device)

    # Training Loop Variables
    best_metric = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)

        # Validate
        val_loss, val_metric = evaluate(model, val_loader, device, loss_fn)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Metric: {val_metric}"
        )

        # Early Stopping & Checkpointing
        # Metric is negative, higher is better
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with metric: {best_metric}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Metric: {best_metric}")
    return best_model_path
