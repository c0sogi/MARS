import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library.utils import MetricMonitor, save_checkpoint
from library.config import Config


def train_one_epoch(model, train_loader, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training data.
        optimizer: Optimizer instance.
        device: Torch device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float).view(-1, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item())

    return metric_monitor.get_avg("Loss")


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: PyTorch model.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Torch device.

    Returns:
        tuple: (Average Loss, AUC Score)
    """
    model.eval()
    metric_monitor = MetricMonitor()
    preds = []
    valid_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float).view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            metric_monitor.update("Loss", loss.item())

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().detach().numpy().flatten())
            valid_targets.extend(targets.cpu().detach().numpy().flatten())

    # Calculate AUC
    # Handle edge cases where validation batch might have only one class (unlikely but safe)
    try:
        auc = roc_auc_score(valid_targets, preds)
    except ValueError:
        auc = 0.5

    return metric_monitor.get_avg("Loss"), auc


def predict(model, test_loader, device):
    """
    Generates predictions for the test set.

    Args:
        model: PyTorch model.
        test_loader: DataLoader for test data.
        device: Torch device.

    Returns:
        tuple: (clip_names array, probabilities array)
    """
    model.eval()
    preds = []
    clips = []

    with torch.no_grad():
        for images, clip_names in test_loader:
            images = images.to(device, dtype=torch.float)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().detach().numpy().flatten())
            clips.extend(clip_names)

    return np.array(clips), np.array(preds)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    save_path_auc,
    save_path_loss,
):
    """
    Orchestrates the full training process with Early Stopping and Checkpointing.
    Saves two checkpoints: one for Best AUC and one for Best Loss.

    Args:
        model: PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Torch device.
        epochs: Total number of epochs.
        patience: Early stopping patience.
        save_path_auc: Path to save the model with best AUC.
        save_path_loss: Path to save the model with best Loss.
    """
    best_auc = 0.0
    best_loss = float("inf")
    early_stopping_counter = 0
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, epochs + 1):
        # --- Training ---
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # --- Scheduler Step ---
        if scheduler is not None:
            scheduler.step()

        # --- Validation ---
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # --- Logging ---
        # Printing full precision as requested
        print(
            f"Epoch: {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # --- Checkpointing ---
        # 1. Best AUC (Discriminator)
        if val_auc > best_auc:
            best_auc = val_auc
            save_checkpoint(model, optimizer, epoch, {"auc": best_auc}, save_path_auc)

        # 2. Best Loss (Calibrator) & Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, {"loss": best_loss}, save_path_loss
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # --- Early Stopping ---
        if early_stopping_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: PyTorch model.
        test_loader: DataLoader for test data.
        device: Torch device.
        output_path: Path to save the submission CSV.
    """
    clips, probs = predict(model, test_loader, device)

    df = pd.DataFrame({"clip": clips, "probability": probs})

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
