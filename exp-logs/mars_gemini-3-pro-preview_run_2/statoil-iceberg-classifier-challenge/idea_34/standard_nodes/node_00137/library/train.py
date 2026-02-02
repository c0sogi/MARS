import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from library.config import (
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    get_model_path,
    BATCH_SIZE,
)
from library.utils import EarlyStopping, save_checkpoint
from library.model import GA_WBN
from library.data_loader import get_dataloaders


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The compute device.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for imgs, angs, lbls in loader:
        imgs = imgs.to(device)
        angs = angs.to(device)
        lbls = lbls.to(device)

        batch_size = imgs.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        # Forward pass
        outputs = model(imgs, angs)
        loss = criterion(outputs, lbls)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: The loss function.
        device: The compute device.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for imgs, angs, lbls in loader:
            imgs = imgs.to(device)
            angs = angs.to(device)
            lbls = lbls.to(device)

            batch_size = imgs.size(0)
            dataset_size += batch_size

            outputs = model(imgs, angs)
            loss = criterion(outputs, lbls)

            running_loss += loss.item() * batch_size

    avg_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return avg_loss


def run_fold_training(fold_idx):
    """
    Runs the training pipeline for a specific fold.

    Args:
        fold_idx (int): The index of the fold to train.

    Returns:
        float: The best validation loss achieved for this fold.
    """
    print(f"Starting training for Fold {fold_idx}...")

    # Get DataLoaders
    # load_cached_data=True is default, but explicit here for clarity
    train_loader, val_loader = get_dataloaders(fold_idx, load_cached_data=True)

    # Initialize Model
    model = GA_WBN().to(DEVICE)

    # Initialize Optimizer (Adam as per design)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Initialize Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Initialize Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Initialize Early Stopping
    model_save_path = get_model_path(fold_idx)
    early_stopping = EarlyStopping(patience=PATIENCE, mode="min", verbose=True)

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = validate(model, val_loader, criterion, DEVICE)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Update Scheduler
        scheduler.step(val_loss)

        # Check Early Stopping
        # EarlyStopping handles saving the checkpoint if the metric improves
        early_stopping(val_loss, model, model_save_path)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold_idx} finished. Best Val Loss: {early_stopping.best_score}")
    return early_stopping.best_score
