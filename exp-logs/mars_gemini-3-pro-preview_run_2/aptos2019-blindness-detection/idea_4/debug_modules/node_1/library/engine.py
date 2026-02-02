import torch
import numpy as np
from library.config import Config
from library.utils import MetricMonitor, quadratic_weighted_kappa


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=None):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The PyTorch model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        criterion (callable): The loss function (e.g., nn.MSELoss).
        optimizer (torch.optim.Optimizer): The optimizer.
        device (torch.device): The device to use for training (CPU or GPU).
        epoch (int, optional): The current epoch number for logging purposes.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        # Model outputs shape (Batch,)
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimization step
        optimizer.step()

        # Update metrics
        metric_monitor.update("Loss", loss.item())

    avg_loss = metric_monitor.get_avg("Loss")

    prefix = f"[Epoch {epoch}] " if epoch is not None else ""
    print(f"{prefix}Train Loss: {avg_loss}")

    return avg_loss


def validate(model, val_loader, criterion, device, epoch=None):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The PyTorch model to evaluate.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        criterion (callable): The loss function.
        device (torch.device): The device to use for evaluation.
        epoch (int, optional): The current epoch number for logging purposes.

    Returns:
        tuple: A tuple containing (average_validation_loss, quadratic_weighted_kappa_score).
    """
    model.eval()
    metric_monitor = MetricMonitor()

    preds = []
    targets_list = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, targets)
            metric_monitor.update("Loss", loss.item())

            # Collect predictions and targets for QWK calculation
            # Move to CPU to avoid GPU memory accumulation
            preds.append(outputs.cpu())
            targets_list.append(targets.cpu())

    avg_loss = metric_monitor.get_avg("Loss")

    # Concatenate all batches
    predictions = torch.cat(preds)
    true_labels = torch.cat(targets_list)

    # Process predictions for QWK (Regression -> Ordinal Labels)
    # 1. Round continuous scores to nearest integer
    # 2. Clamp values to valid class range [0, 4]
    # 3. Convert to integer type
    predicted_labels = torch.round(predictions).int()
    predicted_labels = torch.clamp(predicted_labels, 0, 4)

    # Targets are float (for regression), convert to int for metric calculation
    true_labels_int = true_labels.int()

    # Calculate Quadratic Weighted Kappa
    # We convert tensors to numpy arrays as the metric function expects array-like inputs
    kappa = quadratic_weighted_kappa(true_labels_int.numpy(), predicted_labels.numpy())

    prefix = f"[Epoch {epoch}] " if epoch is not None else ""
    print(f"{prefix}Val Loss: {avg_loss}")
    print(f"{prefix}Val Kappa: {kappa}")

    return avg_loss, kappa
