import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.model import NFWBN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (str): Device to run on ('cpu' or 'cuda').

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for imgs, angles, targets in loader:
        imgs = imgs.to(device)
        angles = angles.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(imgs, angles)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        dataset_size += imgs.size(0)

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for validation data.
        criterion (Loss): The loss function.
        device (str): Device to run on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for imgs, angles, targets in loader:
            imgs = imgs.to(device)
            angles = angles.to(device)
            targets = targets.to(device)

            outputs = model(imgs, angles)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * imgs.size(0)
            dataset_size += imgs.size(0)

    val_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return val_loss


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state.
    """

    def __init__(self, patience=Config.EARLY_STOPPING_PATIENCE, delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves the model state when validation loss decreases."""
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss


def run_fold(fold_idx, train_loader, val_loader):
    """
    Orchestrates the training process for a single cross-validation fold.

    Args:
        fold_idx (int): The index of the current fold.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.

    Returns:
        float: The best validation loss achieved for this fold.
    """
    print(f"Starting training for Fold {fold_idx}...")

    # Initialize model
    model = NFWBN().to(Config.DEVICE)

    # Optimizer (Adam)
    optimizer = optim.Adam(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler (ReduceLROnPlateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=False,  # We handle logging manually if needed, or rely on return values
    )

    # Criterion
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )
        val_loss = validate(model, val_loader, criterion, Config.DEVICE)

        # Step the scheduler
        scheduler.step(val_loss)

        # Print full precision metrics
        print(
            f"Fold {fold_idx} | Epoch {epoch + 1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Check early stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered for Fold {fold_idx} at epoch {epoch + 1}")
            break

    # Save the best model
    save_path = os.path.join(Config.WORK_DIR, f"model_fold_{fold_idx}.pth")
    if early_stopping.best_model_state is not None:
        torch.save(early_stopping.best_model_state, save_path)
        print(f"Best model for Fold {fold_idx} saved to {save_path}")
        return early_stopping.val_loss_min
    else:
        # Fallback if training loop didn't trigger save (e.g. 0 epochs or error)
        # Should not happen in normal execution
        torch.save(model.state_dict(), save_path)
        print(f"Model for Fold {fold_idx} saved to {save_path} (Last Epoch)")
        return val_loss
