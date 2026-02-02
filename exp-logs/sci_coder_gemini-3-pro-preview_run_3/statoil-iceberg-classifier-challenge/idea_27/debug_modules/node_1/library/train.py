import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.data_loader import get_dataloaders
from library.model import HybridSECNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Training data loader.
        criterion (loss_fn): The loss function (BCEWithLogitsLoss).
        optimizer (Optimizer): The optimizer.
        device (str): Device to run on ('cpu' or 'cuda').

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)

        # BCEWithLogitsLoss expects target shape to match output shape (N, 1)
        loss = criterion(outputs, labels.view(-1, 1))

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        loader (DataLoader): Validation data loader.
        criterion (loss_fn): The loss function.
        device (str): Device to run on.

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
            loss = criterion(outputs, labels.view(-1, 1))

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def run_fold(fold_idx):
    """
    Runs the training and validation loop for a specific fold.
    Implements Early Stopping and saves the best model.

    Args:
        fold_idx (int): The index of the fold (0 to N_FOLDS-1).

    Returns:
        float: The best validation loss achieved for this fold.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Starting Fold {fold_idx} on {device}...")

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(fold_idx=fold_idx)

    # Initialize Model
    model = HybridSECNN()
    model.to(device)

    # Optimizer: Adam with constant LR (no scheduler as per strategy)
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function: BCEWithLogitsLoss (numerically stable Log Loss)
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_filename = f"model_fold_{fold_idx}.pth"

    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        epoch_duration = time.time() - epoch_start

        # Print full precision metrics
        print(
            f"Fold {fold_idx} Epoch {epoch+1}/{Config.EPOCHS} "
            f"- Train Loss: {train_loss} "
            f"- Val Loss: {val_loss} "
            f"- Time: {epoch_duration:.2f}s"
        )

        # Check Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model state
            state = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(state, best_model_filename)
            print(
                f"  -> Validation loss improved. Model saved to {best_model_filename}."
            )
        else:
            patience_counter += 1
            print(
                f"  -> Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered for Fold {fold_idx} at epoch {epoch+1}.")
            break

    total_time = time.time() - start_time
    print(
        f"Fold {fold_idx} finished. Best Val Loss: {best_val_loss}. Total Time: {total_time:.2f}s"
    )

    return best_val_loss
