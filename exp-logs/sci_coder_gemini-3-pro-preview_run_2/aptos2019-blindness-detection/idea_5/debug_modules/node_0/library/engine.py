import sys
import time
import numpy as np
import torch
import torch.nn as nn
from library.config import CFG
from library.utils import AverageMeter


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        data_loader (torch.utils.data.DataLoader): The training data loader.
        device (torch.device): The device to run training on.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    # Define Loss function (MSE for regression proxy to QWK)
    criterion = nn.MSELoss()

    for step, (images, labels) in enumerate(data_loader):
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)

        batch_size = images.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        y_preds = model(images)

        # Ensure shapes match for MSE: (B, 1) vs (B,) -> view labels or preds
        loss = criterion(y_preds.view(-1), labels.view(-1))

        # Backward pass
        loss.backward()

        # Gradient clipping
        if CFG.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        # Optimizer step
        optimizer.step()

        # Update metrics
        loss_meter.update(loss.item(), batch_size)

    return loss_meter.avg


def validate(model, data_loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to run evaluation on.

    Returns:
        tuple: (average_loss, predictions, targets)
            - average_loss (float): The average validation loss.
            - predictions (np.ndarray): The raw continuous predictions.
            - targets (np.ndarray): The ground truth labels.
    """
    model.eval()
    loss_meter = AverageMeter()

    criterion = nn.MSELoss()

    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            batch_size = images.size(0)

            # Forward pass
            y_preds = model(images)

            # Calculate loss
            loss = criterion(y_preds.view(-1), labels.view(-1))

            loss_meter.update(loss.item(), batch_size)

            # Collect predictions and targets
            # We keep them continuous for QWK calculation and ensemble averaging later
            preds.append(y_preds.view(-1).detach().cpu().numpy())
            targets.append(labels.view(-1).detach().cpu().numpy())

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(targets)

    return loss_meter.avg, predictions, ground_truth
