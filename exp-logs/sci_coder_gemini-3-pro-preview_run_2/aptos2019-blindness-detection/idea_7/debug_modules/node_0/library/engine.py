import os
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter, quadratic_weighted_kappa


def train_one_epoch(model, train_loader, optimizer, device, scheduler=None):
    """
    Executes one training epoch.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): The training dataloader.
        optimizer (Optimizer): The optimizer.
        device (str or torch.device): The device to use.
        scheduler (optional): Learning rate scheduler. If provided, it is stepped after each batch.

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    # Use Mean Squared Error Loss for regression
    criterion = nn.MSELoss()

    for images, targets in train_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Flatten outputs to match target shape (B,)
        outputs = outputs.view(-1)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if Config.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Scheduler step (if batch-level)
        if scheduler is not None:
            scheduler.step()

        # Update metrics
        loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def valid_one_epoch(model, val_loader, device):
    """
    Executes one validation epoch.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): The validation dataloader.
        device (str or torch.device): The device to use.

    Returns:
        tuple: (average_loss, qwk_score, predictions, targets)
    """
    model.eval()
    loss_meter = AverageMeter()
    criterion = nn.MSELoss()

    preds = []
    targets_list = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            outputs = outputs.view(-1)

            # Compute loss
            loss = criterion(outputs, targets)
            loss_meter.update(loss.item(), images.size(0))

            # Collect predictions and targets
            preds.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    # Concatenate all batches
    preds = np.concatenate(preds)
    targets_list = np.concatenate(targets_list)

    # Calculate Quadratic Weighted Kappa
    # The utility function handles rounding internally for regression outputs
    qwk = quadratic_weighted_kappa(targets_list, preds)

    return loss_meter.avg, qwk, preds, targets_list


class EarlyStopping:
    """
    Early stopping utility to stop training when a monitored metric stops improving.
    """

    def __init__(self, patience=5, mode="max", min_delta=1e-4, save_path=None):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            mode (str): One of {'min', 'max'}.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            save_path (str): Path to save the best model state_dict.
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if mode == "min":
            self.val_score = np.inf
        else:
            self.val_score = -np.inf

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        else:
            if self.mode == "min":
                if score < self.best_score - self.min_delta:
                    self.best_score = score
                    self.save_checkpoint(model)
                    self.counter = 0
                else:
                    self.counter += 1
            else:  # mode == 'max'
                if score > self.best_score + self.min_delta:
                    self.best_score = score
                    self.save_checkpoint(model)
                    self.counter = 0
                else:
                    self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model):
        """Saves model when validation metric improves."""
        if self.save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
