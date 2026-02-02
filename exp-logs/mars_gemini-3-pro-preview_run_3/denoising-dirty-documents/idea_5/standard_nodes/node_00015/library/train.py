import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import gc

from library.config import Config
from library.model import RDN
from library.dataset import prepare_datasets
from library.utils import seed_everything


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass: Predict Noise
        outputs = model(inputs)

        # Loss: MSE between Predicted Noise and Actual Noise (Input - Clean)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # Accumulate loss (multiply by batch size to handle last batch correctly)
        running_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

    average_loss = running_loss / total_samples if total_samples > 0 else 0.0
    return average_loss


def validate(model, dataloader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run validation on.

    Returns:
        tuple: (average_mse_loss, rmse_score)
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    average_mse = running_loss / total_samples if total_samples > 0 else 0.0
    rmse_score = np.sqrt(average_mse)

    return average_mse, rmse_score


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    load_cached_data=True,
    patience=10,
    num_workers=Config.NUM_WORKERS,
):
    """
    Main function to run the training pipeline.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        load_cached_data (bool): Whether to load data from cache.
        patience (int): Early stopping patience.
        num_workers (int): Number of workers for data loading.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Starting training on device: {device}")

    # 2. Data Preparation
    train_dataset, val_dataset = prepare_datasets(load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Model Initialization
    model = RDN().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler: Reduce LR when validation RMSE stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 4. Training Loop
    best_rmse = float("inf")
    early_stop_counter = 0

    print("-" * 60)

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_mse, val_rmse = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step(val_rmse)

        # Logging (Full precision as requested)
        print(
            f"Epoch {epoch + 1}/{epochs} | Train MSE: {train_loss} | Val MSE: {val_mse} | Val RMSE: {val_rmse}"
        )

        # Checkpointing & Early Stopping
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print("-" * 60)
    print(f"Training completed. Best Validation RMSE: {best_rmse}")

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
