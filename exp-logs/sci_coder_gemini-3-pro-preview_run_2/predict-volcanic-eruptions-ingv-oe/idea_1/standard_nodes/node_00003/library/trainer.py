import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random

from library.config import Config
from library.model import VolcanoMLP
from library.data_loader import prepare_data


def set_seed(seed):
    """
    Sets random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs).squeeze()
        # Handle case where batch size is 1 or squeeze removes too many dims
        if outputs.ndim == 0:
            outputs = outputs.unsqueeze(0)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to evaluate on.

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs).squeeze()
            if outputs.ndim == 0:
                outputs = outputs.unsqueeze(0)

            loss = criterion(outputs, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    val_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return val_loss


def run_training(
    debug_size=Config.DEBUG_SAMPLE_SIZE,
    epochs=Config.EPOCHS,
    lr=Config.LEARNING_RATE,
    patience=Config.EARLY_STOPPING_PATIENCE,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    load_cached_data=True,
):
    """
    Orchestrates the full training pipeline.

    Args:
        debug_size (int, optional): Limit dataset size for debugging.
        epochs (int): Maximum number of epochs.
        lr (float): Learning rate.
        patience (int): Early stopping patience.
        batch_size (int): Batch size.
        num_workers (int): Number of dataloader workers.
        load_cached_data (bool): Whether to use cached features.

    Returns:
        model: The trained PyTorch model (best state loaded).
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Preparation
    # Uses library.data_loader which handles caching, scaling, and splitting
    train_loader, val_loader, scaler, input_dim = prepare_data(
        debug_size=debug_size,
        batch_size=batch_size,
        num_workers=num_workers,
        load_cached_data=load_cached_data,
    )

    # 3. Model Initialization
    model = VolcanoMLP(
        input_dim=input_dim,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # 4. Training Components
    criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss (Scaled): {train_loss} - Val Loss (Scaled): {val_loss}"
        )

        # Update Scheduler
        scheduler.step(val_loss)

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Best Validation MAE: {best_val_loss}")

    # Load best model state
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    return model
