import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.utils import MetricMonitor
from library.config import Config


def train_one_epoch(model, train_loader, optimizer, device, epoch):
    """
    Executes one training epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for updating weights.
        device (str): Device to run training on ('cuda' or 'cpu').
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device).view(-1, 1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item())

    print(f"Epoch: {epoch} | Train | {metric_monitor}")
    return metric_monitor.metrics["Loss"]["avg"]


def valid_one_epoch(model, val_loader, device, epoch):
    """
    Executes one validation epoch.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        device (str): Device to run validation on.
        epoch (int): Current epoch number.

    Returns:
        tuple: (AUC score, Average Loss)
    """
    model.eval()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device).view(-1, 1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            # Convert logits to probabilities
            preds = torch.sigmoid(outputs)

            metric_monitor.update("Loss", loss.item())

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Flatten lists of arrays into single 1D arrays
    all_targets = np.concatenate(all_targets).ravel()
    all_preds = np.concatenate(all_preds).ravel()

    # Calculate AUC
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle case where only one class is present in the batch/subset
        auc = 0.5

    metric_monitor.update("AUC", auc)

    print(f"Epoch: {epoch} | Valid | {metric_monitor}")
    return auc, metric_monitor.metrics["Loss"]["avg"]


def tta_inference_fn(model, loader, device):
    """
    Performs inference with Test Time Augmentation (TTA).
    Averages predictions across: Original, H-Flip, V-Flip, HV-Flip.

    Args:
        model (torch.nn.Module): The trained model.
        loader (DataLoader): DataLoader for inference data.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # 1. Original
            outputs = model(images)
            preds = torch.sigmoid(outputs)

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, dims=[3])
            outputs_h = model(images_h)
            preds += torch.sigmoid(outputs_h)

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, dims=[2])
            outputs_v = model(images_v)
            preds += torch.sigmoid(outputs_v)

            # 4. Horizontal + Vertical Flip
            images_hv = torch.flip(images, dims=[2, 3])
            outputs_hv = model(images_hv)
            preds += torch.sigmoid(outputs_hv)

            # Average
            preds /= 4.0

            all_preds.extend(preds.cpu().numpy())

    return np.concatenate(all_preds).ravel()
