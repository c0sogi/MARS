import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import (
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    WORKING_DIR,
)
from library.utils import get_logger, set_seed
from library.data import get_dataloaders
from library.model import SFWIVModel

logger = get_logger(__name__)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        # Ensure targets are (N, 1) float tensors for BCEWithLogitsLoss
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Calculate ROC AUC
        # Handle edge case where only one class is present in the batch
        try:
            auc_score = roc_auc_score(all_targets, all_preds)
        except ValueError:
            auc_score = 0.5
    else:
        auc_score = 0.5

    return epoch_loss, auc_score


def run_training(load_cached_data=True):
    """
    Main function to run the training pipeline.
    """
    set_seed()

    logger.info(f"Starting training on device: {DEVICE}")

    # 1. Get Dataloaders
    train_loader, val_loader = get_dataloaders(load_cached_data=load_cached_data)

    # 2. Initialize Model
    model = SFWIVModel(pretrained=True)
    model = model.to(DEVICE)

    # 3. Define Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        logger.info(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"New best model saved to {best_model_path} with AUC: {best_auc}"
            )
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(
                f"Early stopping triggered after {patience_counter} epochs with no improvement."
            )
            break

    logger.info(f"Training complete. Best Validation AUC: {best_auc}")
    return best_auc
