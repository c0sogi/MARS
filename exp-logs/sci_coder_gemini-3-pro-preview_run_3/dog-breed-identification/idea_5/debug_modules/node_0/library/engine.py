import torch
import torch.nn as nn
import numpy as np
from library.utils import MetricMonitor
from library.config import Config


def train_one_epoch(model, train_loader, optimizer, device, epoch, model_ema=None):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model to train.
        train_loader: DataLoader for the training set.
        optimizer: Optimizer instance.
        device: Device to train on (cuda/cpu).
        epoch: Current epoch number (for logging).
        model_ema: Optional ModelEMA instance for shadow weights update.

    Returns:
        dict: Average metrics (Loss, Accuracy) for the epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()
    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update Model EMA (Shadow Model)
        if model_ema is not None:
            model_ema.update(model)

        # Calculate Accuracy
        with torch.no_grad():
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == targets).sum().item() / targets.size(0)

        # Update metrics
        metric_monitor.update("Loss", loss.item())
        metric_monitor.update("Accuracy", accuracy)

    # Print full precision metrics as required
    print(f"Epoch {epoch} Training: {metric_monitor}")

    return metric_monitor.avg


def valid_one_epoch(model, val_loader, device, epoch):
    """
    Validates the model on the validation set.

    Args:
        model: The PyTorch model to validate.
        val_loader: DataLoader for the validation set.
        device: Device to validate on.
        epoch: Current epoch number (for logging).

    Returns:
        dict: Average metrics (Loss, Accuracy) for the epoch.
    """
    model.eval()
    metric_monitor = MetricMonitor()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Calculate Accuracy
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == targets).sum().item() / targets.size(0)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)

    # Print full precision metrics
    print(f"Epoch {epoch} Validation: {metric_monitor}")

    return metric_monitor.avg


def inference_fn(model, test_loader, device):
    """
    Generates predictions for the test set using Test-Time Augmentation (TTA).
    TTA Strategy: Average probabilities of original and horizontally flipped images.

    Args:
        model: The PyTorch model.
        test_loader: DataLoader for the test set.
        device: Device to run inference on.

    Returns:
        tuple: (predictions, ids)
            predictions: Numpy array of shape (N, Num_Classes) with probabilities.
            ids: List of image IDs corresponding to predictions.
    """
    model.eval()
    final_preds = []
    final_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device, non_blocking=True)

            # 1. Inference on original images
            output_orig = model(images)
            probs_orig = torch.softmax(output_orig, dim=1)

            # 2. Inference on horizontally flipped images (TTA)
            # Flip along width dimension (dim 3 for NCHW tensor)
            images_flipped = torch.flip(images, dims=[3])
            output_flip = model(images_flipped)
            probs_flip = torch.softmax(output_flip, dim=1)

            # 3. Average the probabilities
            avg_probs = (probs_orig + probs_flip) / 2.0

            final_preds.append(avg_probs.cpu().numpy())
            final_ids.extend(ids)

    final_preds = np.concatenate(final_preds)
    return final_preds, final_ids
