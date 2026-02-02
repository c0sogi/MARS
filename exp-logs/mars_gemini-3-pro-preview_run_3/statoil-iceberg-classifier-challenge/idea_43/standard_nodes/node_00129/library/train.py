import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.utils import set_seed, get_device
from library.model import CAFPCNN
from library.data import get_loaders


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        # Unpack batch
        images, angles, targets = batch

        # Move to device
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).view(-1, 1)  # Ensure shape matches logits (B, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(images, angles)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for batch in loader:
            # Unpack batch
            images, angles, targets = batch

            # Move to device
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).view(-1, 1)

            # Forward pass
            logits = model(images, angles)

            # Compute loss
            loss = criterion(logits, targets)

            # Statistics
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def fit_fold(
    fold,
    n_folds=5,
    epochs=75,
    patience=12,
    batch_size=32,
    learning_rate=1e-3,
    weight_decay=1e-4,  # L2 Regularization
    save_dir="./working/idea_43/checkpoints/",
):
    """
    Trains a model for a specific fold with early stopping.
    """
    # Ensure reproducibility
    set_seed(42)

    # Setup directories
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"model_fold_{fold}.pth")

    # Device
    device = get_device()

    # Initialize Model
    model = CAFPCNN()
    model.to(device)

    # Get Data Loaders
    train_loader, val_loader, _ = get_loaders(
        fold=fold,
        n_folds=n_folds,
        batch_size=batch_size,
        num_workers=2,
        load_cached_data=True,
    )

    # Optimizer and Loss
    # AdamW with constant learning rate decouples weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for Fold {fold}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Print full precision metrics
        print(
            f"Fold {fold} - Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Checkpoint and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            # print(f"Validation loss improved. Saved model to {checkpoint_path}")
        else:
            patience_counter += 1
            # print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    print(f"Fold {fold} finished. Best Val Loss: {best_val_loss}")
    return best_val_loss
