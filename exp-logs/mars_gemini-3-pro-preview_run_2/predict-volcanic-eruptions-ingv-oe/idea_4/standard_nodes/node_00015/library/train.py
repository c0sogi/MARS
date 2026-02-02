import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

from library.config import Config
from library.utils import seed_everything, TargetScaler
from library.dataset import VolcanoDataset
from library.model import AttentionPooledHybridEfficientNet


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model: The neural network model.
        dataloader: DataLoader for the training set.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to run calculations on.

    Returns:
        float: Average loss for the epoch (on scaled targets).
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        spectrogram = batch["spectrogram"].to(device)
        features = batch["features"].to(device)
        target = batch["target"].to(device).unsqueeze(1)  # [Batch, 1]

        batch_size = spectrogram.size(0)

        optimizer.zero_grad()

        outputs = model(spectrogram, features)
        loss = criterion(outputs, target)

        loss.backward()

        # Gradient clipping to prevent exploding gradients in RNNs/Deep Nets
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, dataloader, criterion, device, scaler):
    """
    Performs one epoch of validation.

    Args:
        model: The neural network model.
        dataloader: DataLoader for the validation set.
        criterion: Loss function.
        device: Device to run calculations on.
        scaler: TargetScaler instance for inverse transformation.

    Returns:
        tuple: (Average Scaled Loss, Average Original MAE)
    """
    model.eval()
    running_loss = 0.0
    running_mae_original = 0.0
    dataset_size = 0

    with torch.no_grad():
        for batch in dataloader:
            spectrogram = batch["spectrogram"].to(device)
            features = batch["features"].to(device)
            target = batch["target"].to(device).unsqueeze(1)

            batch_size = spectrogram.size(0)

            outputs = model(spectrogram, features)
            loss = criterion(outputs, target)

            running_loss += loss.item() * batch_size

            # Calculate MAE on original scale (Time to Eruption)
            # Inverse transform returns tensors on the same device
            outputs_original = scaler.inverse_transform(outputs)
            target_original = scaler.inverse_transform(target)

            mae_batch = torch.abs(outputs_original - target_original).mean()
            running_mae_original += mae_batch.item() * batch_size

            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    epoch_mae_original = running_mae_original / dataset_size

    return epoch_loss, epoch_mae_original


def run_training(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    debug=Config.DEBUG,
):
    """
    Main training pipeline.

    Args:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate for AdamW.
        debug (bool): If True, uses a small subset of data for debugging.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Preparation
    print("Initializing Target Scaler...")
    target_scaler = TargetScaler()

    print("Loading Datasets...")
    # Train dataset initializes and fits the scaler
    train_dataset = VolcanoDataset(
        metadata_path=Config.TRAIN_METADATA, mode="train", target_scaler=target_scaler
    )

    val_dataset = VolcanoDataset(
        metadata_path=Config.VAL_METADATA, mode="val", target_scaler=target_scaler
    )

    # Handle Debug Mode
    if debug:
        print("Debug mode enabled: Using subset of 100 samples.")
        train_indices = list(range(min(len(train_dataset), 100)))
        val_indices = list(range(min(len(val_dataset), 100)))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Attention-Pooled Hybrid Model...")
    model = AttentionPooledHybridEfficientNet()
    model.to(device)

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss function for optimization (on scaled targets)
    criterion = nn.L1Loss()

    # 5. Training Loop
    best_val_mae = float("inf")
    patience_counter = 0

    print(f"Starting Training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae_original = validate_one_epoch(
            model, val_loader, criterion, device, target_scaler
        )

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss (Scaled): {train_loss} - "
            f"Val Loss (Scaled): {val_loss} - "
            f"Val MAE (Original): {val_mae_original}"
        )

        # Early Stopping & Checkpointing
        # We save based on the original scale MAE as that is the competition metric
        if val_mae_original < best_val_mae:
            best_val_mae = val_mae_original
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(
                f"New best model saved to {Config.MODEL_SAVE_PATH} with Val MAE: {best_val_mae}"
            )
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training completed. Best Validation MAE: {best_val_mae}")
