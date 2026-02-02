import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import AverageMeter, save_checkpoint
from library.model import SimpleCNN


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, device):
    """
    Trains the model for one epoch.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): The neural network model.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        epoch (int): Current epoch number.
        device (torch.device): Device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, angles, targets) in enumerate(train_loader):
        # Move data to the same device as model
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)  # Reshape to [Batch, 1]

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
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
        model (nn.Module): The neural network model.
        criterion (nn.Module): The loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for images, angles, targets in val_loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

    return losses.avg


def fit_fold(fold_idx, train_loader, val_loader):
    """
    Manages the training loop for a specific fold, including early stopping and checkpointing.

    Args:
        fold_idx (int): The index of the current fold.
        train_loader (DataLoader): DataLoader for the training subset.
        val_loader (DataLoader): DataLoader for the validation subset.

    Returns:
        float: The best validation loss achieved for this fold.
    """
    print(f"Starting training for Fold {fold_idx}...")

    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = SimpleCNN().to(device)

    # Initialize Optimizer (Adam with constant LR as per Config)
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Loss Function (BCEWithLogitsLoss for binary classification)
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping and Checkpointing variables
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train for one epoch
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, device
        )

        # Validate
        val_loss = validate(val_loader, model, criterion, device)

        # Print metrics with full precision
        elapsed = time.time() - start_time
        print(
            f"Fold {fold_idx} Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Time: {elapsed}s"
        )

        # Check for improvement
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_score": best_loss,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            fold_idx,
        )

        # Early Stopping check
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered for Fold {fold_idx} at epoch {epoch}.")
            break

    print(f"Fold {fold_idx} finished. Best Val Loss: {best_loss}")
    return best_loss
