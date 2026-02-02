import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import MetricMonitor, get_logger

logger = get_logger()


def mixup_data(x, y, alpha=1.0, device="cpu"):
    """
    Applies Mixup augmentation to inputs and targets.
    Returns mixed inputs and mixed targets.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    # y is expected to be (Batch, 1) float tensor
    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b

    return mixed_x, mixed_y


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.
    """
    metric_monitor = MetricMonitor()
    model.train()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device).view(-1, 1)  # Ensure targets are (Batch, 1)

        # Apply Mixup
        mixed_images, mixed_targets = mixup_data(
            images, targets, Config.MIXUP_ALPHA, device
        )

        # Forward pass
        outputs = model(mixed_images)

        # Loss calculation
        loss = criterion(outputs, mixed_targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item())

    # Log training metrics
    logger.info(f"Epoch {epoch} Train: {metric_monitor}")
    return metric_monitor.get_avg("Loss")


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    metric_monitor = MetricMonitor()
    model.eval()

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device).view(-1, 1)

            # Forward pass (no mixup)
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Calculate Accuracy
            preds = torch.sigmoid(outputs)
            predicted_labels = (preds > 0.5).float()
            accuracy = (predicted_labels == targets).float().mean()

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy.item())

    # Retrieve averages
    avg_loss = metric_monitor.get_avg("Loss")
    avg_acc = metric_monitor.get_avg("Accuracy")

    # Print full precision as requested
    logger.info(f"Val: Loss: {avg_loss}, Accuracy: {avg_acc}")

    return avg_loss, avg_acc
