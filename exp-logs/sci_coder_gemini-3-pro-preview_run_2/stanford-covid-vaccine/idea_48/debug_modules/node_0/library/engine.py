import torch
import torch.nn as nn
import numpy as np
import os
from library.config import Config
from library.loss_metric import MCRMSELoss, GlobalMCRMSE


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model: The PyTorch model (EIPFN).
        loader: DataLoader for training data.
        optimizer: The optimizer.
        criterion: The loss function (MCRMSELoss).
        device: The device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, partner_map, targets) in enumerate(loader):
        inputs = inputs.to(device)
        partner_map = partner_map.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass: returns predictions from both passes
        y_pred_1, y_pred_2 = model(inputs, partner_map)

        # Calculate loss for both passes
        loss_1 = criterion(y_pred_1, targets)
        loss_2 = criterion(y_pred_2, targets)

        # Weighted sum of losses
        loss = (Config.LOSS_WEIGHT_PASS1 * loss_1) + (Config.LOSS_WEIGHT_PASS2 * loss_2)

        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        device: The device to run on.

    Returns:
        float: The Global MCRMSE score.
    """
    model.eval()
    metric = GlobalMCRMSE()

    with torch.no_grad():
        for inputs, partner_map, targets in loader:
            inputs = inputs.to(device)
            partner_map = partner_map.to(device)
            targets = targets.to(device)

            # Forward pass
            # We only care about the refined prediction (Pass 2) for validation
            _, y_pred_2 = model(inputs, partner_map)

            # Update metric accumulator
            metric.update(y_pred_2, targets)

    return metric.compute()


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    patience=Config.PATIENCE,
):
    """
    Main training loop with early stopping and model saving.

    Args:
        model: The PyTorch model.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        num_epochs: Maximum number of epochs.
        device: Device to run on.
        patience: Patience for early stopping.
    """
    criterion = MCRMSELoss()
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")
    print(f"Epochs: {num_epochs}, Patience: {patience}")

    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_score)
            else:
                scheduler.step()

        # Print metrics (Full precision)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.2e} | Train Loss: {train_loss:.8f} | Val MCRMSE: {val_score:.15f}"
        )

        # Early Stopping and Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MCRMSE: {best_score:.15f}")
