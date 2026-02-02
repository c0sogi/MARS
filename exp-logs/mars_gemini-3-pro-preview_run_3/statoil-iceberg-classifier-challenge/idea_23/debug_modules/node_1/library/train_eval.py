import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import AverageMeter, save_checkpoint, log_message, set_seed
from library.model import MSMANet
from library.data_loader import get_loaders


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): The model to train.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (str): Device to use (cpu or cuda).
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, angles, labels) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images, angles)

        # Reshape labels to match output shape (B, 1)
        labels = labels.view(-1, 1)

        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): The model to evaluate.
        criterion (nn.Module): Loss function.
        device (str): Device to use.

    Returns:
        float: Average loss (Log Loss) on the validation set.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for images, angles, labels in val_loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            labels = labels.view(-1, 1)

            loss = criterion(outputs, labels)
            losses.update(loss.item(), images.size(0))

    return losses.avg


def train_fold(fold_idx):
    """
    Trains the model for a specific fold using the configuration parameters.
    Implements Early Stopping and saves the best model.

    Args:
        fold_idx (int): The fold index to train (0 to NUM_FOLDS-1).

    Returns:
        float: The best validation loss achieved for this fold.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    log_message(f"Starting training for Fold {fold_idx}...")

    # Get DataLoaders
    train_loader, val_loader, _ = get_loaders(fold_idx=fold_idx, debug=Config.DEBUG)

    # Initialize Model
    model = MSMANet()
    model = model.to(device)

    # Loss and Optimizer
    # BCEWithLogitsLoss combines Sigmoid and BCELoss, numerically stable
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Training Loop variables
    best_val_loss = float("inf")
    patience_counter = 0

    # Directory for saving checkpoints for this fold
    fold_save_dir = Config.WORKING_DIR
    checkpoint_filename = f"checkpoint_fold_{fold_idx}.pth"
    best_filename = f"model_best_fold_{fold_idx}.pth"

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss = validate(val_loader, model, criterion, device)

        duration = time.time() - start_time

        # Print metrics with full precision
        log_message(
            f"Fold {fold_idx} | Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | "
            f"Time: {duration:.2f}s"
        )

        # Early Stopping and Checkpointing
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            log_message(f"  New best validation loss: {best_val_loss}")
        else:
            patience_counter += 1
            log_message(
                f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "fold": fold_idx,
                "config": {
                    "model": Config.MODEL_NAME,
                    "lr": Config.LEARNING_RATE,
                    "batch_size": Config.BATCH_SIZE,
                },
            },
            is_best,
            save_dir=fold_save_dir,
            filename=checkpoint_filename,
            best_filename=best_filename,
        )

        if patience_counter >= Config.PATIENCE:
            log_message(f"Early stopping triggered at epoch {epoch+1}")
            break

    log_message(f"Fold {fold_idx} finished. Best Val Loss: {best_val_loss}")
    return best_val_loss
