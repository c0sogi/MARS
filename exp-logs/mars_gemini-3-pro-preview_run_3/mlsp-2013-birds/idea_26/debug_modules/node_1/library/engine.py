import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import calculate_robust_auc


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The model to train (CNN or MLP).
        dataloader (DataLoader): The training dataloader (returns mixed inputs/targets).
        optimizer (Optimizer): The optimizer (Adam/AdamW).
        scheduler (LRScheduler): The learning rate scheduler.
        device (str): Computation device ('cpu' or 'cuda').
        epoch (int): Current epoch number (for logging/scheduler).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    # Iterate over the dataloader
    # Note: Progress bars are suppressed as per requirements
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        # Targets are already mixed if MixupCollate was used
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()

    # Step the scheduler at the end of the epoch
    if scheduler is not None:
        scheduler.step()

    avg_loss = running_loss / len(dataloader)

    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The validation dataloader.
        device (str): Computation device.

    Returns:
        tuple: (avg_loss, auc_score, predictions, targets)
            - avg_loss (float): Average validation loss.
            - auc_score (float): Robust ROC AUC score.
            - predictions (np.ndarray): Predicted probabilities.
            - targets (np.ndarray): Ground truth labels.
    """
    model.eval()
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            # Store predictions and targets for metric calculation
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / len(dataloader)

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
    else:
        all_preds = np.array([])
        all_targets = np.array([])

    # Calculate robust AUC
    auc_score = calculate_robust_auc(all_targets, all_preds)

    return avg_loss, auc_score, all_preds, all_targets
