import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import roc_auc_score

from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    MODEL_SAVE_PATH,
    LEARNING_RATE,
    EPOCHS,
    SEED,
    DEVICE,
)
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import HRVANet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (Batch, 1)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    return running_loss / count if count > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    auc = 0.5
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Check if we have more than one class to calculate AUC
        if len(np.unique(all_targets)) > 1:
            auc = roc_auc_score(all_targets, all_preds)

    return avg_loss, auc


def fit(epochs=EPOCHS, load_cached_data=True):
    """
    Main training loop with early stopping and model saving.
    """
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Loading
    # The caching logic is handled within get_dataloaders via library/data_loader.py
    train_loader, val_loader, _ = get_dataloaders(
        TRAIN_META_PATH,
        VAL_META_PATH,
        TEST_META_PATH,
        load_cached_data=load_cached_data,
    )

    if train_loader is None or val_loader is None:
        print("Error: Could not load training or validation data.")
        return

    # 3. Model Initialization
    model = HRVANet().to(device)

    # 4. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")
