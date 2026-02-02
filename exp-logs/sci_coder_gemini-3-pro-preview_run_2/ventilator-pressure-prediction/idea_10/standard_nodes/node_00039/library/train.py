import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import math

from library.config import Config
from library.utils import seed_everything, WeightedL1Loss
from library.dataset import load_and_preprocess_data
from library.model import RGIBiLSTM


def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        X = batch["X"].to(device)
        u_out = batch["u_out"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(X)

        # Calculate weighted loss
        loss = criterion(preds, y, u_out)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate_one_epoch(model, loader, device):
    """
    Performs validation. Calculates MAE strictly on the inspiratory phase (u_out == 0).
    """
    model.eval()
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            u_out = batch["u_out"].to(device)
            y = batch["y"].to(device)

            preds = model(X)

            # Calculate MAE only for inspiratory phase (u_out == 0)
            # Create boolean mask where u_out is 0
            mask = u_out == 0

            # Filter predictions and targets
            valid_preds = preds[mask]
            valid_targets = y[mask]

            if len(valid_targets) > 0:
                mae_sum = torch.abs(valid_preds - valid_targets).sum().item()
                total_mae += mae_sum
                total_count += len(valid_targets)

    if total_count == 0:
        return 0.0

    return total_mae / total_count


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data(
        load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = RGIBiLSTM().to(device)

    # 4. Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_MIN_LR
    )

    criterion = WeightedL1Loss()

    # 5. Training Loop
    best_val_mae = float("inf")
    early_stopping_patience = 30
    epochs_no_improve = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config.MAX_GRAD_NORM
        )

        # Validate
        val_mae = validate_one_epoch(model, val_loader, device)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val MAE: {val_mae} | "
            f"LR: {current_lr:.8f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing
        if val_mae < best_val_mae:
            print(
                f"Validation MAE improved from {best_val_mae} to {val_mae}. Saving model..."
            )
            best_val_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early Stopping
        if epochs_no_improve >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epochs_no_improve} epochs without improvement."
            )
            break

    print(f"Training complete. Best Validation MAE: {best_val_mae}")
