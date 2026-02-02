import os
import time
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.losses import HybridLoss


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): Optimizer instance.
        criterion (nn.Module): Loss function.
        device (str): Device to run training on ('cuda' or 'cpu').

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks in dataloader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device, threshold=0.5):
    """
    Evaluates the model on the validation set using Global Dice Coefficient.

    Global Dice is computed by summing intersections and unions across the entire
    dataset, rather than averaging per-sample Dice scores.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation set.
        device (str): Device to run evaluation on.
        threshold (float): Threshold for binarizing predictions.

    Returns:
        float: Global Dice score.
    """
    model.eval()
    total_intersection = 0.0
    total_union = 0.0
    epsilon = 1e-6

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            # Flatten tensors to compute global stats
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            # Accumulate Intersection and Union
            intersection = (preds_flat * masks_flat).sum().item()
            union = preds_flat.sum().item() + masks_flat.sum().item()

            total_intersection += intersection
            total_union += union

    # Compute Global Dice
    global_dice = (2.0 * total_intersection) / (total_union + epsilon)
    return global_dice


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs=Config.EPOCHS,
    patience=10,
):
    """
    Main training loop with Early Stopping and Best Model Checkpointing.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (str): Device to run on.
        num_epochs (int): Maximum number of epochs.
        patience (int): Epochs to wait for improvement before early stopping.

    Returns:
        float: Best validation Dice score achieved.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Initialize Loss
    # Using HybridLoss as specified in the idea (BCE + BatchDice)
    criterion = HybridLoss(bce_weight=1.0, dice_weight=1.0)

    best_dice = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    print(f"Starting training on device: {device}")
    print(f"Epochs: {num_epochs}, Patience: {patience}")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_dice = validate(model, val_loader, device, threshold=Config.THRESHOLD)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        elapsed_time = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Time: {elapsed_time:.2f}s | "
            f"LR: {current_lr} | "
            f"Train Loss: {train_loss} | "
            f"Val Dice: {val_dice}"
        )

        # Checkpointing and Early Stopping
        if val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New best model saved! Dice improved to {best_dice}")
        else:
            epochs_no_improve += 1
            print(f"  >>> No improvement. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    print(f"Training complete. Best Global Dice: {best_dice} at Epoch {best_epoch + 1}")
    return best_dice
