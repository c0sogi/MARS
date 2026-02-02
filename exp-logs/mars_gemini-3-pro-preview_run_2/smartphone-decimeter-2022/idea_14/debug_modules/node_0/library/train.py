import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import get_logger, set_seed
from library.data_loader import load_data, GNSSWindowDataset
from library.model import SkyStateTransformer

# Initialize logger
logger = get_logger("train")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to run training on.

    Returns:
        Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_seq, batch_sky, batch_y in dataloader:
        batch_seq = batch_seq.to(device)
        batch_sky = batch_sky.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_seq, batch_sky)

        # Compute loss
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_seq.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run validation on.

    Returns:
        Average loss for the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_seq, batch_sky, batch_y in dataloader:
            batch_seq = batch_seq.to(device)
            batch_sky = batch_sky.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_seq, batch_sky)
            loss = criterion(outputs, batch_y)

            running_loss += loss.item() * batch_seq.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached=True):
    """
    Orchestrates the entire training process.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoaders.
        load_cached (bool): Whether to load cached data or process from scratch.

    Returns:
        float: Best validation loss achieved.
    """
    # Set random seed for reproducibility
    set_seed(Config.RANDOM_STATE)

    logger.info("Initializing training process...")

    # 1. Load Data
    # load_data handles caching logic internally based on the flag
    logger.info(f"Loading data (Cached: {load_cached})...")
    (train_data, val_data, _) = load_data(load_cached_data=load_cached)

    train_X_seq, train_X_sky, train_y = train_data
    val_X_seq, val_X_sky, val_y, _ = val_data

    # 2. Create Datasets and DataLoaders
    train_dataset = GNSSWindowDataset(train_X_seq, train_X_sky, train_y)
    val_dataset = GNSSWindowDataset(val_X_seq, val_X_sky, val_y)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    logger.info(
        f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
    )

    # 3. Initialize Model and Hardware
    device = torch.device(Config.DEVICE)
    model = SkyStateTransformer().to(device)

    # 4. Define Loss, Optimizer, and Scheduler
    criterion = nn.L1Loss()  # Mean Absolute Error
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info("Starting training loop...")

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Log metrics (full precision)
        logger.info(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
        )

        # Update Scheduler
        scheduler.step(val_loss)

        # Checkpointing and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_PATH)
            logger.info(
                f"New best model saved to {Config.MODEL_PATH} with Val Loss: {best_val_loss}"
            )
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss}"
                )
                break

    logger.info("Training completed.")
    return best_val_loss
