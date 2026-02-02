import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import set_seed
from library.data_loader import get_loaders
from library.model import WA_IDPH_CNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): The training data loader.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        device (str): Device to train on ('cuda' or 'cpu').

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, angles, labels) in enumerate(loader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass: Model accepts both image and angle
        outputs = model(images, angles)

        # BCEWithLogitsLoss expects labels to be same shape as outputs (float)
        # outputs are (B,), labels are (B,)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): The validation data loader.
        criterion (nn.Module): The loss function.
        device (str): Device to evaluate on.

    Returns:
        float: Average validation loss (Log Loss).
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def run_fold(fold_idx):
    """
    Runs the training and validation loop for a specific fold.
    Implements Early Stopping and saves the best model.

    Args:
        fold_idx (int): The index of the current fold (0-4).

    Returns:
        float: The best validation loss achieved for this fold.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Starting Fold {fold_idx} on {device}...")

    # Initialize Model
    model = WA_IDPH_CNN()
    model.to(device)

    # Get Data Loaders
    train_loader, val_loader = get_loaders(fold_idx)

    # Define Loss and Optimizer
    # BCEWithLogitsLoss is numerically stable for Log Loss
    criterion = nn.BCEWithLogitsLoss()

    # AdamW with constant learning rate (no scheduler as per design)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")

    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        print(
            f"Fold {fold_idx} | Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.10f} | "
            f"Time: {epoch_duration:.2f}s"
        )

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (Val Loss: {val_loss:.10f})")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    total_time = time.time() - start_time
    print(
        f"Fold {fold_idx} finished in {total_time:.2f}s. Best Val Loss: {best_val_loss:.10f}"
    )

    return best_val_loss
