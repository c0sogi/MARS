import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import MetricMonitor


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch using BCEWithLogitsLoss.

    Args:
        model: PyTorch model.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler (optional).
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number (for printing).

    Returns:
        dict: Dictionary containing 'Loss' and 'AUC' for the epoch.
    """
    model.train()
    monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    all_targets = []
    all_preds = []

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape (B, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        # Note: Scheduler step is typically handled per-epoch outside this function
        # or per-step here depending on the scheduler type.
        # Given Config parameters, we assume per-epoch stepping in the main loop.

        # Update Loss metric
        monitor.update("Loss", loss.item())

        # Store predictions for global AUC calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.detach().cpu().numpy())

    # Compute Global AUC for the epoch
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    monitor.update("AUC", epoch_auc)

    # Print metrics with full precision
    print(f"Epoch {epoch} Train: {monitor}")

    return monitor.get_metrics()


def validate_with_tta(model, loader, device):
    """
    Validates the model using 4-view Test Time Augmentation (TTA).
    Views: Original, Horizontal Flip, Vertical Flip, Horizontal+Vertical Flip.

    Args:
        model: PyTorch model.
        loader: DataLoader for validation data.
        device: 'cuda' or 'cpu'.

    Returns:
        dict: Dictionary containing 'Loss' and 'AUC'.
    """
    model.eval()
    monitor = MetricMonitor()

    # Since we average probabilities from TTA, we use BCELoss for validation tracking
    criterion = nn.BCELoss()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            # --- TTA Strategy ---
            # 1. Original View
            logits_1 = model(images)
            probs_1 = torch.sigmoid(logits_1)

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, [3])
            logits_2 = model(images_h)
            probs_2 = torch.sigmoid(logits_2)

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, [2])
            logits_3 = model(images_v)
            probs_3 = torch.sigmoid(logits_3)

            # 4. Horizontal + Vertical Flip
            images_hv = torch.flip(images, [2, 3])
            logits_4 = model(images_hv)
            probs_4 = torch.sigmoid(logits_4)

            # Aggregate: Average probabilities
            avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

            # Compute Loss
            loss = criterion(avg_probs, targets)
            monitor.update("Loss", loss.item())

            # Store for AUC
            all_preds.append(avg_probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Compute Global AUC
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    try:
        val_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        val_auc = 0.5

    monitor.update("AUC", val_auc)

    print(f"Validation: {monitor}")

    return monitor.get_metrics()


def predict_with_tta(model, loader, device):
    """
    Generates predictions for a dataset using 4-view TTA.
    Useful for generating test set predictions for the ensemble.

    Args:
        model: PyTorch model.
        loader: DataLoader (typically test set).
        device: 'cuda' or 'cpu'.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)

            # TTA Views
            probs_1 = torch.sigmoid(model(images))
            probs_2 = torch.sigmoid(model(torch.flip(images, [3])))  # H-Flip
            probs_3 = torch.sigmoid(model(torch.flip(images, [2])))  # V-Flip
            probs_4 = torch.sigmoid(model(torch.flip(images, [2, 3])))  # HV-Flip

            # Average
            avg_probs = (probs_1 + probs_2 + probs_3 + probs_4) / 4.0

            all_preds.append(avg_probs.cpu().numpy())

    return np.concatenate(all_preds)
