import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import log_loss
from library.config import Config
from library.utils import AverageMeter, save_checkpoint
from library.model import MicroResNet


def train_one_epoch(train_loader, model, criterion, optimizer, epoch):
    """
    Executes one training epoch.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): The model to train.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average training loss for the epoch.
    """
    losses = AverageMeter()
    model.train()

    for i, (images, angles, target) in enumerate(train_loader):
        # Move data to configured device
        images = images.to(Config.DEVICE)
        angles = angles.to(Config.DEVICE)
        target = target.to(Config.DEVICE).view(-1, 1)

        # Forward pass
        output = model(images, angles)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(val_loader, model, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): The model to evaluate.
        criterion (nn.Module): The loss function.

    Returns:
        tuple: (average_bce_loss, sklearn_log_loss)
    """
    losses = AverageMeter()
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (images, angles, target) in enumerate(val_loader):
            images = images.to(Config.DEVICE)
            angles = angles.to(Config.DEVICE)
            target = target.to(Config.DEVICE).view(-1, 1)

            # Forward pass
            output = model(images, angles)
            loss = criterion(output, target)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities for Log Loss calculation
            preds = torch.sigmoid(output).cpu().numpy()
            targets = target.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets)

    # Calculate Log Loss using sklearn to match competition metric precisely
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # Calculate log loss (eps handles numerical stability, labels ensures correct class handling)
    try:
        val_log_loss = log_loss(y_true, y_pred, eps=1e-15, labels=[0, 1])
    except Exception as e:
        print(
            f"Warning: Sklearn log_loss calculation failed ({e}). Defaulting to BCE loss."
        )
        val_log_loss = losses.avg

    return losses.avg, val_log_loss


def train_fold(fold, train_loader, val_loader):
    """
    Manages the full training loop for a single fold.

    Args:
        fold (int): The current fold index.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.

    Returns:
        float: The best validation log loss achieved for this fold.
    """
    print(f"[{Config.PROJECT_NAME}] Starting training for Fold {fold}...")

    # Initialize Model
    model = MicroResNet().to(Config.DEVICE)

    # Initialize Optimizer (Adam with constant Learning Rate)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function (BCEWithLogitsLoss combines Sigmoid and BCE)
    criterion = nn.BCEWithLogitsLoss()

    # State tracking
    best_log_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(train_loader, model, criterion, optimizer, epoch)

        # Validate
        val_loss, val_log_loss = validate(val_loader, model, criterion)

        # Print metrics with full precision
        print(
            f"Fold {fold} | Epoch {epoch + 1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Log Loss: {val_log_loss}"
        )

        # Checkpointing logic
        is_best = val_log_loss < best_log_loss

        if is_best:
            best_log_loss = val_log_loss
            patience_counter = 0
            # Save best model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_log_loss": best_log_loss,
                    "fold": fold,
                },
                is_best=True,
                fold=fold,
            )
        else:
            patience_counter += 1
            # Save checkpoint (overwrite generic fold checkpoint)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_log_loss": best_log_loss,
                    "fold": fold,
                },
                is_best=False,
                fold=fold,
            )

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Fold {fold} | Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Fold {fold} | Finished. Best Log Loss: {best_log_loss}")
    return best_log_loss
