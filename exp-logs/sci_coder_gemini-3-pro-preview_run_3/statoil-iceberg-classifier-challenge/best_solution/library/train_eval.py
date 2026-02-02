import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
    SEED,
)
from library.utils import set_seed, get_logger
from library.data_loader import get_dataloaders
from library.model import IDPH_CNN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch: forward pass, loss calculation, backward pass, and optimization.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (str): The device to run on ('cpu' or 'cuda').

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): The validation data loader.
        criterion (Loss): The loss function.
        device (str): The device to run on.

    Returns:
        float: The average validation loss.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    avg_loss = running_loss / total_samples
    return avg_loss


def run_fold(fold_idx, load_cached_data=True):
    """
    Orchestrates the training process for a single fold.

    Args:
        fold_idx (int): The index of the current fold (0-based).
        load_cached_data (bool): Whether to load pre-processed data from cache.

    Returns:
        float: The best validation loss achieved for this fold.
    """
    logger = get_logger(f"Fold_{fold_idx}")
    set_seed(SEED)

    logger.info(f"Starting training for Fold {fold_idx}")

    # Retrieve DataLoaders for the specific fold
    train_loader, val_loader = get_dataloaders(
        fold_idx=fold_idx, load_cached=load_cached_data
    )

    # Initialize Model
    model = IDPH_CNN().to(DEVICE)

    # Initialize Optimizer (AdamW with constant LR) and Loss Function
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Setup Checkpointing
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"model_fold_{fold_idx}.pth")

    best_val_loss = float("inf")
    patience_counter = 0

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        # Log metrics with full precision
        logger.info(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    logger.info(f"Fold {fold_idx} completed. Best Validation Loss: {best_val_loss}")
    return best_val_loss
