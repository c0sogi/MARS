import os
import torch
import torch.nn as nn
import numpy as np
import gc
from torch.cuda.amp import autocast, GradScaler
from library.config import Config


def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler=None, scheduler=None
):
    """
    Trains the model for one epoch.

    Args:
        model: The neural network.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss function.
        device: Device to run on.
        scaler: GradScaler for mixed precision.
        scheduler: Learning rate scheduler (optional, if stepped per batch).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    # Use provided scaler or create a dummy one if not provided (though A100 should use it)
    if scaler is None:
        scaler = GradScaler()

    for batch_idx, batch_data in enumerate(loader):
        # Unpack data
        images = batch_data["image"].to(device, non_blocking=True)
        targets = batch_data["targets"].to(device, non_blocking=True)
        patient_targets = batch_data["patient_target"].to(device, non_blocking=True)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            logits = model(images)
            loss = criterion(logits, targets, patient_targets)

        # Backward pass with scaler
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        # Clean up
        del images, targets, patient_targets, logits, loss

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The neural network.
        loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to run on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for batch_data in loader:
            images = batch_data["image"].to(device, non_blocking=True)
            targets = batch_data["targets"].to(device, non_blocking=True)
            patient_targets = batch_data["patient_target"].to(device, non_blocking=True)

            batch_size = images.size(0)

            # Forward pass (no autocast needed for validation usually, but safe to use)
            # Using autocast here ensures consistency with training precision
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets, patient_targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            del images, targets, patient_targets, logits, loss

    val_loss = running_loss / dataset_size
    return val_loss


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    epochs=Config.EPOCHS,
    patience=3,
    save_path=Config.MODEL_SAVE_PATH,
):
    """
    Main training loop with Early Stopping.

    Args:
        model: The neural network.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        criterion: Loss function.
        scheduler: Learning rate scheduler.
        device: Device.
        epochs: Maximum number of epochs.
        patience: Epochs to wait for improvement before stopping.
        save_path: Path to save the best model.
    """
    best_val_loss = float("inf")
    patience_counter = 0
    scaler = GradScaler()

    print(f"Starting training on device: {device}")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{epochs} | LR: {current_lr} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Model saved to {save_path}")
        else:
            patience_counter += 1
            print(
                f"No improvement in validation loss. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        # Garbage collection to prevent OOM
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Training complete. Best Validation Loss: {best_val_loss}")

    # Load best model weights before returning
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model
