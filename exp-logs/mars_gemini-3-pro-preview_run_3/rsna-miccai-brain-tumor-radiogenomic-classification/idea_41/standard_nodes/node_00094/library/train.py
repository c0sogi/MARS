import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from library.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    MODEL_SAVE_PATH,
    seed_everything,
    setup_directories,
)
from library.utils import get_device, calculate_roc_auc
from library.data_loader import get_dataloaders
from library.model import SiameseEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The Siamese Neural Network.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Computation device.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for inputs, targets in loader:
        # Unpack Siamese inputs
        x_even, x_odd = inputs

        # Move to device
        x_even = x_even.to(device)
        x_odd = x_odd.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape (B, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(x_even, x_odd)

        # Loss calculation
        loss = criterion(logits, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        count += targets.size(0)

    return running_loss / count if count > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The Siamese Neural Network.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Computation device.

    Returns:
        tuple: (Average validation loss, ROC AUC score)
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            x_even, x_odd = inputs
            x_even = x_even.to(device)
            x_odd = x_odd.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(x_even, x_odd)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * targets.size(0)
            count += targets.size(0)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    if len(all_targets) > 0:
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        auc_score = calculate_roc_auc(all_targets, all_probs)
    else:
        auc_score = 0.5

    return avg_loss, auc_score


def run_training(load_cached_data=True):
    """
    Orchestrates the training process including setup, training loop,
    validation, early stopping, and model saving.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # 1. Setup
    seed_everything()
    setup_directories()
    device = get_device()
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=BATCH_SIZE, load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    print("Initializing SiameseEfficientNet...")
    model = SiameseEfficientNet()
    model.to(device)

    # 4. Optimization
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # 5. Training Loop
    best_val_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Logging (Full precision)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_val_auc:
            print(
                f"Validation AUC improved from {best_val_auc} to {val_auc}. Saving model..."
            )
            best_val_auc = val_auc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement in Validation AUC. Patience: {patience_counter}/{PATIENCE}"
            )

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_val_auc}")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
