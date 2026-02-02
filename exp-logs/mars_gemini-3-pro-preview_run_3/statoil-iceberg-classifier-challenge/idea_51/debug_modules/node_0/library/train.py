import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import AverageMeter, save_checkpoint, log_loss_score
from library.model import DPDCNN


def train_one_fold(fold, train_loader, val_loader):
    """
    Trains the DPDCNN model for a single fold.

    Args:
        fold (int): The current fold index.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.

    Returns:
        tuple: (best_model_state, best_val_preds, best_val_targets)
    """
    # Device configuration
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = DPDCNN()
    model.to(device)

    # Loss Function and Optimizer
    # We use BCEWithLogitsLoss for numerical stability with the raw logits output by the model
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Tracking variables for Early Stopping and Best Model
    best_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_preds = None
    best_targets = None

    print(f"Starting training for Fold {fold}...")

    for epoch in range(Config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = AverageMeter()

        for images, angles, targets in train_loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).view(-1, 1)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), images.size(0))

        # --- Validation Phase ---
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for images, angles, targets in val_loader:
                images = images.to(device)
                angles = angles.to(device)
                targets = targets.to(device).view(-1, 1)

                outputs = model(images, angles)

                # Apply sigmoid to get probabilities for metric calculation
                probs = torch.sigmoid(outputs)

                val_preds_list.append(probs.cpu().numpy())
                val_targets_list.append(targets.cpu().numpy())

        # Concatenate predictions and targets for the entire validation set
        val_preds = np.concatenate(val_preds_list).ravel()
        val_targets = np.concatenate(val_targets_list).ravel()

        # Calculate Validation Log Loss (Competition Metric)
        val_log_loss = log_loss_score(val_targets, val_preds)

        # Print metrics with full precision
        print(
            f"Fold {fold} | Epoch {epoch + 1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss.avg} | "
            f"Val Log Loss: {val_log_loss}"
        )

        # --- Checkpointing & Early Stopping ---
        is_best = val_log_loss < best_loss

        # Save checkpoint
        # This saves 'checkpoint_fold_X.pth' every epoch, and copies to 'model_best_fold_X.pth' if is_best
        save_checkpoint(model.state_dict(), is_best, fold)

        if is_best:
            best_loss = val_log_loss
            patience_counter = 0
            # Cache the best state and predictions to return later
            best_state = model.state_dict()
            best_preds = val_preds
            best_targets = val_targets
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered for Fold {fold} at Epoch {epoch + 1}. Best Loss: {best_loss}"
            )
            break

    # Ensure we return valid data even if loop finishes without early stopping
    if best_state is None:
        best_state = model.state_dict()
        best_preds = val_preds
        best_targets = val_targets

    return best_state, best_preds, best_targets
