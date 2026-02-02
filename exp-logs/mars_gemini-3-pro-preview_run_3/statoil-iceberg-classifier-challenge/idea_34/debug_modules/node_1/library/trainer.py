import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, calculate_log_loss, get_device
from library.model import SDHAResNet
from library.dataset import get_data_loaders


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (torch.device): Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model expects (x, angle)
        logits = model(images, angles)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for validation data.
        criterion (Loss): The loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (average_loss, log_loss_score)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            logits = model(images, angles)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            num_batches += 1

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        score = calculate_log_loss(all_targets, all_preds)
    else:
        score = 0.0

    return avg_loss, score


def train_fold(fold_idx, load_cached_data=True):
    """
    Orchestrates the training for a specific fold.

    Args:
        fold_idx (int): Index of the fold to train.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        float: Best validation log loss achieved.
    """
    seed_everything(Config.SEED)
    device = get_device()

    print(f"Starting training for Fold {fold_idx} on device: {device}")

    # 1. Data Loading
    train_loader, val_loader, _ = get_data_loaders(
        fold_idx=fold_idx, load_cached_data=load_cached_data
    )

    # 2. Model Initialization
    model = SDHAResNet()
    model.to(device)

    # 3. Optimizer and Loss
    # AdamW with constant learning rate and weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop with Early Stopping
    best_val_log_loss = float("inf")
    patience_counter = 0

    # Ensure checkpoint directory exists
    checkpoint_dir = Config.WORKING_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_fold_{fold_idx}.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_log_loss = validate(model, val_loader, criterion, device)

        print(
            f"Fold {fold_idx} | Epoch {epoch + 1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Val Log Loss: {val_log_loss}"
        )

        # Check for improvement
        if val_log_loss < best_val_log_loss:
            best_val_log_loss = val_log_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), checkpoint_path)
            # print(f"  New best model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            # print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Fold {fold_idx} finished. Best Val Log Loss: {best_val_log_loss}")
    return best_val_log_loss
