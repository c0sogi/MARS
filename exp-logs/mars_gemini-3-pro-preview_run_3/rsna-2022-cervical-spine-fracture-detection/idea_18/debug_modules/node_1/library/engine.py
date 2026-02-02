import torch
import torch.nn as nn
import numpy as np
import os
import sys
from library.config import Config
from library.utils import seed_everything


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: The training dataloader.
        optimizer: The optimizer.
        loss_fn: The loss function.
        device: The device to train on.
        epoch: Current epoch number (for logging).

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        # Inputs shape: (Batch, Seq, Channels, H, W)
        # Targets shape: (Batch, 8)
        logits = model(inputs)

        loss = loss_fn(logits, targets)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, loss_fn, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: The validation dataloader.
        loss_fn: The loss function.
        device: The device to evaluate on.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(inputs)

            loss = loss_fn(logits, targets)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs=Config.EPOCHS,
    patience=3,
    save_path=None,
):
    """
    Main training loop with Early Stopping and Scheduler.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        loss_fn: Loss function.
        device: Device.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model. If None, uses Config.WORKING_DIR/best_model.pth.
    """
    if save_path is None:
        save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")
    print(f"Epochs: {epochs}, Patience: {patience}, Batch Size: {Config.BATCH_SIZE}")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )

        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        print(
            f"Epoch {epoch}/{epochs} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f}"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"Validation loss improved. Model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss:.8f}")
