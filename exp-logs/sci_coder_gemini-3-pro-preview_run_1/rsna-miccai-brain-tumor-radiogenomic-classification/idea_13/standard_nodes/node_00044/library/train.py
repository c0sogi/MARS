import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, get_roc_auc, setup_logger
from library.data import get_dataloaders
from library.model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device, label_smoothing=0.0):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Apply Label Smoothing manually
        # y_smooth = y * (1 - epsilon) + 0.5 * epsilon
        if label_smoothing > 0:
            targets_smooth = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing
        else:
            targets_smooth = targets

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # BCEWithLogitsLoss expects targets to be same shape as logits (N, 1)
        loss = criterion(logits, targets_smooth.unsqueeze(1))

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    """
    Performs validation on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            logits = model(images)
            loss = criterion(logits, targets.unsqueeze(1))

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(logits).squeeze(1)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    auc_score = get_roc_auc(all_targets, all_probs)

    return epoch_loss, auc_score


def run_fold(load_cached_data=True):
    """
    Orchestrates the training process.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # Setup
    logger = setup_logger()
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    logger.info(f"Starting training on device: {device}")

    # Data Loading
    train_loader, val_loader, _, _ = get_dataloaders(load_cached_data=load_cached_data)

    # Model Initialization
    model = build_model(device=device)

    # Loss and Optimizer
    # Note: We handle label smoothing manually in train_one_epoch, so we use standard BCE here
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Training Loop Variables
    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0

    # Ensure save directory exists
    save_dir = os.path.dirname(os.path.join(Config.CACHE_DIR, "best_model.pth"))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pth")

    logger.info("Starting Training Loop...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            label_smoothing=Config.LABEL_SMOOTHING,
        )

        # Validate
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Time: {elapsed:.2f}s - "
            f"Train Loss: {train_loss:.8f} - "
            f"Val Loss: {val_loss:.8f} - "
            f"Val AUC: {val_auc:.16f}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved with AUC: {best_val_auc:.16f}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info(
        f"Training complete. Best Validation AUC: {best_val_auc:.16f} at Epoch {best_epoch+1}"
    )
    return best_val_auc
