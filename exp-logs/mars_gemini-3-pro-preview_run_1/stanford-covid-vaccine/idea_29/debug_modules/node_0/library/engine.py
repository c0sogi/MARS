import torch
import torch.nn as nn
import numpy as np
import os
import sys
from library.config import Config
from library.utils import MCRMSE, get_device


def train_one_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer.
        device (torch.device): Compute device.

    Returns:
        float: Average training loss.
    """
    model.train()
    running_loss = 0.0
    criterion = nn.MSELoss()

    # Iterate over batches
    for batch_idx, (seq, loop, dist, targets) in enumerate(dataloader):
        # Move to device
        seq = seq.to(device)
        loop = loop.to(device)
        dist = dist.to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(seq, loop, dist)

        # Masking: Calculate loss only for the first 68 positions
        # outputs shape: (Batch, 107, 3) -> slice to (Batch, 68, 3)
        # targets shape: (Batch, 107, 3) -> slice to (Batch, 68, 3)
        outputs_scored = outputs[:, : Config.SEQ_SCORED, :]
        targets_scored = targets[:, : Config.SEQ_SCORED, :]

        # Compute Loss (MSE)
        loss = criterion(outputs_scored, targets_scored)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using MCRMSE.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        device (torch.device): Compute device.

    Returns:
        float: MCRMSE score.
    """
    model.eval()
    mcrmse_metric = MCRMSE()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for seq, loop, dist, targets in dataloader:
            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)

            # Forward pass
            outputs = model(seq, loop, dist)

            # Collect data on CPU to compute metric over the full set
            # Slice to scored positions (0-67)
            all_preds.append(outputs[:, : Config.SEQ_SCORED, :].cpu())
            all_targets.append(targets[:, : Config.SEQ_SCORED, :].cpu())

    # Concatenate all batches
    if len(all_preds) == 0:
        return 0.0

    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE
    score = mcrmse_metric(y_true, y_pred)

    return score.item()


def train_and_evaluate(model, train_loader, val_loader, patience=5):
    """
    Main training loop with validation, early stopping, and checkpointing.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        patience (int): Number of epochs to wait for improvement before early stopping.
    """
    device = get_device()
    model.to(device)

    # Optimizer: AdamW with low weight decay for RNN stability
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    best_mcrmse = float("inf")
    epochs_no_improve = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.10f} | "
            f"LR: {current_lr:.2e}"
        )

        # Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            epochs_no_improve = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  >>> New Best Model Saved (Score: {best_mcrmse:.10f})")
        else:
            epochs_no_improve += 1

        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse:.10f}")


def get_predictions(model, dataloader):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): The trained model.
        dataloader (DataLoader): Test data loader.

    Returns:
        tuple: (predictions tensor, list of ids)
    """
    device = get_device()
    model.to(device)
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for seq, loop, dist, sample_ids in dataloader:
            seq = seq.to(device)
            loop = loop.to(device)
            dist = dist.to(device)

            # Forward pass
            outputs = model(seq, loop, dist)

            all_preds.append(outputs.cpu())
            all_ids.extend(sample_ids)

    predictions = torch.cat(all_preds, dim=0)
    return predictions, all_ids
