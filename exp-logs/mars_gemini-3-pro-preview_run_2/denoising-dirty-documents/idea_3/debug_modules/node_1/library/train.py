import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

from library.config import Config
from library.dataset import DenoisingDataset
from library.model import ResUNetPlusPlus
from library.utils import set_seed, get_device, calculate_rmse


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch using Deep Supervision.
    """
    model.train()
    running_loss = 0.0

    for noisy, residual in dataloader:
        noisy = noisy.to(device)
        residual = residual.to(device)

        optimizer.zero_grad()

        # Forward pass
        # With Deep Supervision, outputs is a list: [out1, out2, out3, out4]
        outputs = model(noisy)

        # Calculate Multi-Scale Loss
        loss = 0
        # Ensure outputs is a list/tuple for iteration (safety check)
        if isinstance(outputs, (list, tuple)):
            for output in outputs:
                loss += criterion(output, residual)
        else:
            loss = criterion(outputs, residual)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, device):
    """
    Validates the model on the validation set.
    Returns the RMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for noisy, residual in dataloader:
            noisy = noisy.to(device)

            # Forward pass
            # In eval mode, model returns only the final output tensor
            output = model(noisy)

            # Move to CPU for metric calculation
            all_preds.append(output.cpu())
            all_targets.append(residual)

    # Concatenate all batches
    if len(all_preds) > 0:
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)
    else:
        return 0.0

    # Calculate RMSE
    # RMSE(Predicted_Residual, True_Residual) is equivalent to RMSE(Predicted_Clean, True_Clean)
    rmse = calculate_rmse(y_true, y_pred)
    return rmse


def run_training(
    epochs: int = Config.EPOCHS,
    batch_size: int = Config.BATCH_SIZE,
    learning_rate: float = Config.LEARNING_RATE,
    weight_decay: float = Config.WEIGHT_DECAY,
    debug: bool = Config.DEBUG,
    load_cached_data: bool = Config.LOAD_CACHED_DATA,
):
    """
    Main training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()

    # 2. Data Preparation
    train_dataset = DenoisingDataset(
        metadata_file=Config.TRAIN_METADATA,
        mode="train",
        patches_per_image=Config.PATCHES_PER_IMAGE,
        load_cached_data=load_cached_data,
    )

    val_dataset = DenoisingDataset(
        metadata_file=Config.VAL_METADATA, mode="val", load_cached_data=load_cached_data
    )

    # Debug mode: Use a small subset
    if debug:
        train_indices = range(
            min(len(train_dataset), Config.DEBUG_SAMPLES * Config.PATCHES_PER_IMAGE)
        )
        val_indices = range(min(len(val_dataset), Config.DEBUG_SAMPLES))
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 3. Model, Optimizer, Scheduler
    model = ResUNetPlusPlus().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = nn.MSELoss()

    # 4. Training Loop
    best_rmse = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_rmse = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Logging (Full precision)
        print(
            f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss} | Val RMSE: {val_rmse}"
        )

        # Early Stopping & Checkpointing
        if val_rmse < best_rmse - Config.MIN_DELTA:
            best_rmse = val_rmse
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_PATH)
            # print(f"New best model saved with RMSE: {best_rmse}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Training complete. Best Val RMSE: {best_rmse}")
