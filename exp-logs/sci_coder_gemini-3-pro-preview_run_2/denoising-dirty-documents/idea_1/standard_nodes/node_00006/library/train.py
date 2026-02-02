import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import set_seed
from library.model import UNet
from library.dataset import load_processed_data, DenoisingDataset


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()

        # Accumulate loss weighted by batch size
        running_loss += loss.item() * noisy.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set and calculates RMSE.
    """
    model.eval()
    mse_sum = 0.0
    total_pixels = 0

    with torch.no_grad():
        for batch in loader:
            # Validation loader returns (noisy, clean, id)
            if len(batch) == 3:
                noisy, clean, _ = batch
            else:
                continue

            noisy = noisy.to(device)
            clean = clean.to(device)

            outputs = model(noisy)

            # Calculate squared error for this image
            diff = (outputs - clean) ** 2
            mse_sum += diff.sum().item()
            total_pixels += clean.numel()

    if total_pixels == 0:
        return float("inf")

    rmse = np.sqrt(mse_sum / total_pixels)
    return rmse


def run_training(
    load_cached_data=True, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE
):
    """
    Orchestrates the training process including data loading, model setup,
    training loop, and early stopping.
    """
    set_seed()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load Data
    # Uses the provided dataset module which handles caching logic
    train_data = load_processed_data(
        Config.TRAIN_METADATA_PATH, "train", load_cached_data
    )
    val_data = load_processed_data(Config.VAL_METADATA_PATH, "val", load_cached_data)

    # Datasets
    train_dataset = DenoisingDataset(train_data, mode="train")
    val_dataset = DenoisingDataset(val_data, mode="val")

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Validation uses batch_size=1 to handle variable image sizes
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model Setup
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler to reduce LR when validation metric plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_val_rmse = float("inf")
    patience_counter = 0

    # Ensure model save directory exists
    os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_rmse = validate(model, val_loader, device)

        # Step the scheduler
        scheduler.step(val_rmse)

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss (MSE): {train_loss} | Val RMSE: {val_rmse}"
        )

        # Early Stopping and Model Checkpointing
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! RMSE: {val_rmse}")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val RMSE: {best_val_rmse}")
