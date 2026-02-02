import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.model import NBHACNN
from library.data import get_loaders
from library.utils import AverageMeter, calculate_log_loss, set_seed


def train_one_epoch(train_loader, model, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        train_loader: DataLoader for training data.
        model: The neural network model.
        optimizer: The optimizer.
        criterion: The loss function.
        device: The device to run on (cpu or cuda).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, angles, targets in train_loader:
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device)

        # Forward pass
        logits = model(images, angles)
        loss = criterion(logits, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(val_loader, model, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        val_loader: DataLoader for validation data.
        model: The neural network model.
        criterion: The loss function.
        device: The device to run on.

    Returns:
        tuple: (average_loss, log_loss_metric)
    """
    model.eval()
    losses = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, targets in val_loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(images, angles)
            loss = criterion(logits, targets)

            # Update loss meter
            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities for metric calculation
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    # Calculate Log Loss metric
    # Note: y_true is (N, 1), y_pred is (N, 1). Flatten for sklearn log_loss if needed,
    # though calculate_log_loss wrapper handles array-likes.
    metric = calculate_log_loss(y_true, y_pred)

    return losses.avg, metric


def run_fold(config: Config, fold_idx: int):
    """
    Runs the training and validation loop for a specific cross-validation fold.

    Args:
        config (Config): Configuration object.
        fold_idx (int): The index of the current fold (0-based).
    """
    # Set seed for reproducibility
    set_seed(config.seed + fold_idx)

    print(f"Starting Fold {fold_idx}")

    # Initialize Model
    model = NBHACNN(config)
    model = model.to(config.device)

    # Initialize Optimizer
    # Strategy: AdamW with constant LR, decoupling weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Get DataLoaders
    train_loader, val_loader, _ = get_loaders(config, fold_idx=fold_idx)

    # Early Stopping Variables
    best_metric = float("inf")
    patience_counter = 0
    best_model_path = config.get_checkpoint_path(fold_idx)

    for epoch in range(config.epochs):
        # Training Step
        train_loss = train_one_epoch(
            train_loader, model, optimizer, criterion, config.device
        )

        # Validation Step
        val_loss, val_metric = validate(val_loader, model, criterion, config.device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Log Loss: {val_metric}"
        )

        # Early Stopping Logic
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), best_model_path)
            print(
                f"New best model saved for fold {fold_idx} with Log Loss: {best_metric}"
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    print(f"Fold {fold_idx} finished. Best Log Loss: {best_metric}")
