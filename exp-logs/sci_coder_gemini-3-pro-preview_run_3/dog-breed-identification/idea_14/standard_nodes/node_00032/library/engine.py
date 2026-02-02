import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import calculate_metric


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (torch.utils.data.DataLoader): Training data loader.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number (for logging/debugging).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    total_loss = 0.0
    num_batches = len(data_loader)

    # Use CrossEntropyLoss which combines LogSoftmax and NLLLoss
    criterion = nn.CrossEntropyLoss()

    # Scaler for Mixed Precision Training
    scaler = torch.amp.GradScaler("cuda")

    for batch_idx, data in enumerate(data_loader):
        images = data["image"].to(device, non_blocking=True)
        labels = data["label"].to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Context
        with torch.amp.autocast("cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): Validation data loader.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (average_loss, predictions, true_labels)
            - average_loss (float): The Multi Class Log Loss on the validation set.
            - predictions (np.ndarray): Predicted probabilities (N, num_classes).
            - true_labels (np.ndarray): Ground truth labels (N,).
    """
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for data in data_loader:
            images = data["image"].to(device, non_blocking=True)
            labels = data["label"].to(device, non_blocking=True)

            # Forward pass
            # We use mixed precision for inference as well for consistency and speed
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item()

            # Apply Softmax to get probabilities
            probs = torch.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_preds, axis=0)
    true_labels = np.concatenate(all_labels, axis=0)

    # Calculate metric using the utility function (Log Loss)
    # Note: We calculate the metric on the full set to be precise,
    # rather than averaging batch losses, although CrossEntropyLoss average is usually close.
    # The competition metric is Log Loss.
    metric_loss = calculate_metric(true_labels, predictions)

    return metric_loss, predictions, true_labels
