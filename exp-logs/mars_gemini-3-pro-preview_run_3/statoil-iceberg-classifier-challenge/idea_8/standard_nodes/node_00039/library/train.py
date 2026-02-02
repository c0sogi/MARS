import time
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, save_checkpoint, AverageMeter
from library.data_loader import get_train_val_loaders
from library.model import ACResNet


def train_one_epoch(train_loader, model, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        train_loader: DataLoader for training data.
        model: The ACResNet model.
        criterion: The loss function (BCEWithLogitsLoss).
        optimizer: The optimizer.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number.

    Returns:
        avg_loss: The average training loss for this epoch.
    """
    model.train()
    losses = AverageMeter()

    for i, (images, angles, labels) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        # Forward pass
        # Model expects (x, angle)
        outputs = model(images, angles)

        # Ensure labels match output shape (B, 1)
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
        val_loader: DataLoader for validation data.
        model: The ACResNet model.
        criterion: The loss function.
        device: 'cuda' or 'cpu'.

    Returns:
        avg_loss: The average validation loss (Log Loss).
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


def run_fold(fold_index, load_cached_data=True):
    """
    Runs the training and validation loop for a specific fold.

    Args:
        fold_index (int): The fold to train.
        load_cached_data (bool): Whether to load pre-processed numpy arrays from cache.
    """
    print(f"Starting training for Fold {fold_index}")

    # Set seed for reproducibility
    set_seed(Config.SEED + fold_index)

    # Device
    device = torch.device(Config.DEVICE)

    # Data Loaders
    train_loader, val_loader = get_train_val_loaders(
        fold_index, load_cached_data=load_cached_data
    )

    # Model
    model = ACResNet()
    model = model.to(device)

    # Criterion (Loss)
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    # Using Adam with L2 Regularization (Weight Decay) as specified in the solution
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Training Loop Variables
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, model, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss = validate(val_loader, model, criterion, device)

        epoch_time = time.time() - epoch_start

        # Print metrics (Full precision)
        print(
            f"Fold {fold_index} | Epoch {epoch + 1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | "
            f"Time: {epoch_time:.2f}s"
        )

        # Checkpoint and Early Stopping
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            print(
                f"New best model found for Fold {fold_index} with Val Loss: {best_val_loss}"
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        # Save checkpoint
        # We save every epoch's state as 'checkpoint.pth' and update 'model_best.pth' if is_best
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            fold_index,
        )

        # Early Stopping Trigger
        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered for Fold {fold_index} at epoch {epoch + 1}"
            )
            break

    total_time = time.time() - start_time
    print(
        f"Fold {fold_index} finished. Best Val Loss: {best_val_loss}. Total Time: {total_time:.2f}s"
    )
