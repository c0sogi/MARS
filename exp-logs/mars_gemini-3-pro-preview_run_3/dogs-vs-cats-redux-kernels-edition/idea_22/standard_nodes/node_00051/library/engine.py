import os
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import get_logger, calculate_log_loss

# Initialize logger
logger = get_logger("engine")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): The training data loader.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): The optimizer.
        device (str): Device to run training on ('cpu' or 'cuda').
        epoch (int): Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape (N, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        loader (DataLoader): The validation data loader.
        criterion (nn.Module): The loss function.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, accuracy, all_logits, all_labels)
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            num_batches += 1

            # Store logits and labels for metric calculation
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate results
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    # Calculate Accuracy
    preds = (torch.sigmoid(torch.from_numpy(all_logits)) > 0.5).float().numpy()
    accuracy = (preds == all_labels).mean()

    return avg_loss, accuracy, all_logits, all_labels


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    patience,
    save_path,
):
    """
    Orchestrates the training process with Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (str): Device.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        save_path (str): Path to save the best model checkpoint.

    Returns:
        tuple: (best_model, best_val_loss, oof_logits, oof_labels)
    """
    criterion = nn.BCEWithLogitsLoss()
    best_val_loss = float("inf")
    patience_counter = 0

    # To store OOF predictions from the best epoch
    best_oof_logits = None
    best_oof_labels = None

    logger.info(f"Starting training for {epochs} epochs with patience {patience}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, val_logits, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        # Log full precision metrics
        logger.info(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Acc: {val_acc}"
        )

        if scheduler:
            scheduler.step()

        # Early Stopping and Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_oof_logits = val_logits
            best_oof_labels = val_labels
            patience_counter = 0

            torch.save(model.state_dict(), save_path)
            logger.info(f"Validation loss improved. Model saved to {save_path}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # Load best model weights
    logger.info(f"Loading best model from {save_path}")
    model.load_state_dict(torch.load(save_path))

    return model, best_val_loss, best_oof_logits, best_oof_labels


def inference_with_tta(model, loader, device):
    """
    Performs inference using Test Time Augmentation (Horizontal Flip).

    Args:
        model (nn.Module): The trained model.
        loader (DataLoader): Data loader (Test or Val).
        device (str): Device.

    Returns:
        tuple: (averaged_logits, targets)
               targets will be IDs (if test) or labels (if val).
    """
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)

            # 1. Forward pass with original images
            logits_orig = model(images)

            # 2. Forward pass with horizontally flipped images (TTA)
            # Flip along width dimension (dim=3 for N,C,H,W)
            images_flipped = torch.flip(images, dims=[3])
            logits_flip = model(images_flipped)

            # 3. Average raw logits
            # Averaging logits is generally more stable than averaging probabilities
            avg_logits = (logits_orig + logits_flip) / 2.0

            all_logits.append(avg_logits.cpu().numpy())

            # targets can be labels (float) or IDs (int/long)
            all_targets.extend(targets.numpy())

    return np.concatenate(all_logits), np.array(all_targets)
