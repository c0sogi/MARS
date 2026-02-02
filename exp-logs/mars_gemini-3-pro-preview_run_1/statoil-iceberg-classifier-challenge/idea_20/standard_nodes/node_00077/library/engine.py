import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import MetricMonitor


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function (BCEWithLogitsLoss).
        device: The device to run on (cpu or cuda).
        epoch: Current epoch number (for logging).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, (images, angles, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        angles = angles.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Apply Label Smoothing manually
        # y_smooth = y * (1 - epsilon) + 0.5 * epsilon
        smooth_labels = (
            labels * (1 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
        )

        optimizer.zero_grad()

        # Forward pass (model handles symmetry averaging internally)
        logits = model(images, angles)

        # Compute loss
        loss = criterion(logits, smooth_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item())

    avg_loss = metric_monitor.metrics["Loss"]["avg"]
    print(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")
    return avg_loss


def evaluate(model, val_loader, criterion, device, epoch=None):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        val_loader: DataLoader for validation data.
        criterion: The loss function (BCEWithLogitsLoss).
        device: The device to run on.
        epoch: Current epoch number (optional, for logging).

    Returns:
        float: Average validation loss (Log Loss).
    """
    model.eval()
    metric_monitor = MetricMonitor()

    with torch.no_grad():
        for batch_idx, (images, angles, labels) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            angles = angles.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images, angles)

            # For validation, we use the true labels (Log Loss)
            loss = criterion(logits, labels)

            metric_monitor.update("Loss", loss.item())

    avg_loss = metric_monitor.metrics["Loss"]["avg"]
    if epoch is not None:
        print(f"Epoch {epoch} | Val Loss: {avg_loss:.6f}")
    else:
        print(f"Val Loss: {avg_loss:.6f}")

    return avg_loss


def swa_step(swa_model, model):
    """
    Updates the SWA model parameters.

    Args:
        swa_model: The AveragedModel instance.
        model: The current training model.
    """
    swa_model.update_parameters(model)


def predict(model, test_loader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    Uses 4 views: Original, FlipLR, FlipUD, Rotate180 (FlipLR + FlipUD).

    Args:
        model: The PyTorch model.
        test_loader: DataLoader for test data.
        device: The device to run on.

    Returns:
        np.array: Flattened array of probabilities.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch[0].to(device, non_blocking=True)
            angles = batch[1].to(device, non_blocking=True)

            # TTA Views
            # 1. Original
            logits_1 = model(images, angles)

            # 2. Horizontal Flip
            images_lr = torch.flip(images, dims=[3])
            logits_2 = model(images_lr, angles)

            # 3. Vertical Flip
            images_ud = torch.flip(images, dims=[2])
            logits_3 = model(images_ud, angles)

            # 4. Rotate 180 (Horizontal + Vertical Flip)
            images_rot180 = torch.flip(images, dims=[2, 3])
            logits_4 = model(images_rot180, angles)

            # Average Probabilities (not logits, to be safe with sigmoid non-linearity)
            probs_1 = torch.sigmoid(logits_1)
            probs_2 = torch.sigmoid(logits_2)
            probs_3 = torch.sigmoid(logits_3)
            probs_4 = torch.sigmoid(logits_4)

            avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

            predictions.append(avg_probs.cpu().numpy())

    # Concatenate all batches: (N, 1) -> (N,)
    return np.concatenate(predictions).flatten()


def generate_submission(model, test_loader, test_ids, device):
    """
    Generates predictions and saves them to a CSV file.

    Args:
        model: The PyTorch model.
        test_loader: DataLoader for test data.
        test_ids: Array of test IDs corresponding to the loader.
        device: The device to run on.
    """
    print("Generating predictions for submission...")
    probs = predict(model, test_loader, device)

    # Create DataFrame
    df = pd.DataFrame({"id": test_ids, "is_iceberg": probs})

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
