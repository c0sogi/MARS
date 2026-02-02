import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import set_seed, get_device
from library.data_loader import get_loaders
from library.model import MSD_SE_CNN


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, angles, targets in loader:
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        # Model expects (images, angles)
        logits = model(images, angles)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, angles, targets in loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            logits = model(images, angles)
            loss = criterion(logits, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    val_loss = running_loss / dataset_size
    return val_loss


def run_fold(
    fold_idx, train_loader, val_loader, device, epochs, patience, checkpoint_dir
):
    """
    Trains a model for a single fold with early stopping.

    Args:
        fold_idx (int): Index of the current fold.
        train_loader (DataLoader): Loader for training.
        val_loader (DataLoader): Loader for validation.
        device (torch.device): Device to run on.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        checkpoint_dir (str): Directory to save model checkpoints.

    Returns:
        float: Best validation loss achieved for this fold.
    """
    print(f"Starting Fold {fold_idx}")

    # Initialize model
    model = MSD_SE_CNN().to(device)

    # Optimizer: Adam with constant LR and Weight Decay (L2)
    # LR = 1e-3, Weight Decay = 1e-4 (standard regularization)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Loss: BCEWithLogitsLoss (combines Sigmoid and BCELoss for stability)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(checkpoint_dir, f"model_best_fold_{fold_idx}.pth")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Print full precision metrics as required
        print(
            f"Fold {fold_idx} Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1} for fold {fold_idx}"
                )
                break

    print(f"Fold {fold_idx} finished. Best Val Loss: {best_val_loss}")
    return best_val_loss


def train_kfold(
    epochs=100,
    batch_size=32,
    patience=10,
    seed=42,
    debug=False,
    cache_dir="./working/idea_21",
):
    """
    Orchestrates the 5-Fold Cross-Validation training pipeline.

    Args:
        epochs (int): Maximum epochs per fold.
        batch_size (int): Batch size.
        patience (int): Early stopping patience.
        seed (int): Random seed.
        debug (bool): If True, runs on a small subset.
        cache_dir (str): Directory for caching processed data and checkpoints.

    Returns:
        list: List of best validation losses for each fold.
    """
    set_seed(seed)
    device = get_device()

    # Ensure checkpoint directory exists
    checkpoint_dir = os.path.join(cache_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Get DataLoaders (handles caching and splitting)
    # n_splits is fixed to 5 as per strategy
    fold_loaders = get_loaders(
        batch_size=batch_size, n_splits=5, seed=seed, debug=debug, cache_dir=cache_dir
    )

    cv_scores = []

    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        best_loss = run_fold(
            fold_idx, train_loader, val_loader, device, epochs, patience, checkpoint_dir
        )
        cv_scores.append(best_loss)

    avg_cv_score = np.mean(cv_scores)
    print(f"Cross-Validation Complete. Average Log Loss: {avg_cv_score}")

    return cv_scores
