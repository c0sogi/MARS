import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import compute_mcc
from library.loss import FocalLoss
from library.model import KinematicMLP


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Training loop for a single epoch.

    Args:
        model: The DCN model.
        dataloader: Training DataLoader.
        criterion: Loss function (FocalLoss).
        optimizer: Optimizer (AdamW).
        device: 'cuda' or 'cpu'.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (features, targets) in enumerate(dataloader):
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass (returns logits)
        logits = model(features)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * features.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Validation loop.

    Args:
        model: The DCN model.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: 'cuda' or 'cpu'.

    Returns:
        tuple: (avg_loss, probabilities, true_labels)
    """
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)

            logits = model(features)
            loss = criterion(logits, targets)

            running_loss += loss.item() * features.size(0)

            # Convert logits to probabilities for metric calculation
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / len(dataloader.dataset)

    # Concatenate all batches
    if len(all_probs) > 0:
        all_probs = np.concatenate(all_probs)
        all_targets = np.concatenate(all_targets)
    else:
        all_probs = np.array([])
        all_targets = np.array([])

    return avg_loss, all_probs, all_targets


def optimize_threshold(y_true, y_probs):
    """
    Finds the best decision threshold maximizing MCC.

    Args:
        y_true: Ground truth labels.
        y_probs: Predicted probabilities.

    Returns:
        tuple: (best_threshold, best_mcc)
    """
    best_mcc = -1.0
    best_thresh = 0.5

    # Search space: 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 99)

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        score = compute_mcc(y_true, y_pred)

        if score > best_mcc:
            best_mcc = score
            best_thresh = thresh

    return best_thresh, best_mcc


def train_model(train_loader, val_loader):
    """
    Main training routine with Early Stopping.

    Args:
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.

    Returns:
        model: The trained model with best weights loaded.
    """
    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    # Initialize Model
    model = KinematicMLP().to(device)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Focal Loss handles class imbalance
    criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

    # Early Stopping variables
    best_val_mcc = -float("inf")
    patience_counter = 0
    best_model_path = Config.MODEL_PATH
    best_threshold_path = Config.THRESHOLD_PATH

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Training Step
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validation Step
        val_loss, val_probs, val_targets = validate(
            model, val_loader, criterion, device
        )

        # Threshold Optimization for current epoch
        # We optimize threshold on validation set to monitor realistic performance
        curr_thresh, curr_mcc = optimize_threshold(val_targets, val_probs)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val MCC: {curr_mcc:.8f} | "
            f"Best Thresh: {curr_thresh:.4f}"
        )

        # Early Stopping Check
        if curr_mcc > best_val_mcc:
            best_val_mcc = curr_mcc
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), best_model_path)

            # Save best threshold
            np.save(best_threshold_path, np.array([curr_thresh]))
            print(f"  -> New best model saved! MCC: {best_val_mcc:.8f}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")

    # Load best weights before returning
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")

    return model
