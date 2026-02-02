import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.data import prepare_datasets
from library.model import PITHNet


def masked_mae_loss(y_pred, y_true, u_out):
    """
    Computes Mean Absolute Error (MAE) only for the inspiratory phase.

    Args:
        y_pred: Predicted pressure (Batch, Sequence_Length)
        y_true: Actual pressure (Batch, Sequence_Length)
        u_out: Expiratory valve status (Batch, Sequence_Length).
               0 = Inspiratory (include), 1 = Expiratory (exclude).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # Create mask: 1 where u_out == 0 (Inspiratory), 0 otherwise
    mask = 1 - u_out

    # Calculate absolute error
    abs_err = torch.abs(y_pred - y_true)

    # Apply mask
    masked_err = abs_err * mask

    # Calculate mean over valid elements
    # Add a small epsilon to denominator to prevent division by zero (though unlikely in batches)
    loss = masked_err.sum() / (mask.sum() + 1e-8)

    return loss


def train_epoch(model, loader, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        u_out = batch["u_out"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(x)

        # Compute loss
        loss = masked_mae_loss(preds, y, u_out)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate_epoch(model, loader, device):
    """
    Runs validation on the provided loader.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            u_out = batch["u_out"].to(device)

            preds = model(x)
            loss = masked_mae_loss(preds, y, u_out)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def run_training(epochs=None, batch_size=None, load_cached_data=True):
    """
    Main driver function for training the PITH-Net model.

    Args:
        epochs (int, optional): Number of epochs to train. Defaults to Config.EPOCHS.
        batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Resolve hyperparameters
    num_epochs = epochs if epochs is not None else Config.EPOCHS

    print(f"Starting training on {device}...")
    print(
        f"Epochs: {num_epochs}, Batch Size: {batch_size if batch_size else Config.BATCH_SIZE}"
    )

    # 2. Prepare Data
    train_loader, val_loader, _, _ = prepare_datasets(
        load_cached_data=load_cached_data, batch_size=batch_size
    )

    # 3. Initialize Model
    model = PITHNet().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    early_stopping_patience = 15  # Stop if no improvement for 15 epochs
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_loss = validate_epoch(model, val_loader, device)

        # Step Scheduler
        scheduler.step(val_loss)

        # Log metrics (Full precision)
        print(
            f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")
    return best_val_loss
