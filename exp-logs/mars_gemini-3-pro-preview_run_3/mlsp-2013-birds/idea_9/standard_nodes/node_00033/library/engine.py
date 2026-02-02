import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import compute_metric


def mixup_data(x, y, alpha=0.4, device="cuda"):
    """
    Applies Mixup to input data and labels.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch using Mixup regularization.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training dataloader.
        criterion (nn.Module): Loss function (e.g., BCEWithLogitsLoss).
        optimizer (Optimizer): Optimizer.
        device (str): Device to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, labels, _ in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Apply Mixup at the bag level
        inputs, targets_a, targets_b, lam = mixup_data(
            inputs, labels, alpha=0.4, device=device
        )

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute Loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataset_size += inputs.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): Validation dataloader.
        criterion (nn.Module): Loss function.
        device (str): Device to run on.

    Returns:
        tuple: (Average validation loss, ROC AUC score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            dataset_size += inputs.size(0)

            # Apply sigmoid to get probabilities for metric calculation
            preds = torch.sigmoid(outputs)

            all_targets.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_targets = np.concatenate(all_targets, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    # Compute Metric (Macro ROC AUC)
    auc_score = compute_metric(all_targets, all_preds)

    return epoch_loss, auc_score
