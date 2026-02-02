import os
import torch
import torch.nn as nn
import torch.optim as optim
import time

from library.config import (
    DEVICE,
    NUM_EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    CHECKPOINT_DIR,
)
from library.utils import AverageMeter, get_logger
from library.model import MAPCNN

logger = get_logger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (str): Device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, angles, labels in loader:
        # Move data to device
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        # Forward pass
        # Model expects (images, angles)
        logits = model(images, angles)

        # Compute loss
        loss = criterion(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str): Device to run on.

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            logits = model(images, angles)
            loss = criterion(logits, labels)

            losses.update(loss.item(), images.size(0))

    return losses.avg


def train_fold(fold_idx, train_loader, val_loader):
    """
    Executes the training pipeline for a single fold, including model initialization,
    training loop, validation, early stopping, and checkpointing.

    Args:
        fold_idx (int): The index of the current fold.
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.

    Returns:
        float: The best validation loss achieved for this fold.
    """
    logger.info(f"Starting training for Fold {fold_idx}...")

    # Initialize Model
    model = MAPCNN()
    model = model.to(DEVICE)

    # Loss Function: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer: Adam with Weight Decay (L2 Regularization)
    # Using 1e-4 for weight decay as suggested by "strong regularization" requirement
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()

        # Train and Validate
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        elapsed = time.time() - start_time

        # Log metrics with full precision
        logger.info(
            f"Fold {fold_idx} | Epoch {epoch}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    logger.info(f"Fold {fold_idx} finished. Best Val Loss: {best_val_loss}")
    return best_val_loss
