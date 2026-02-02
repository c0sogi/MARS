import os
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import log_loss
from library.utils import get_logger, get_checkpoint_path
from library.config import DEVICE

# Initialize logger
logger = get_logger("engine")


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): Training dataloader.
        optimizer (Optimizer): PyTorch optimizer.
        device (str): Device to use ('cuda' or 'cpu').
        epoch (int): Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape [Batch, 1]

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): Validation dataloader.
        device (str): Device to use.

    Returns:
        tuple: (average_loss, log_loss_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate Log Loss
    # Clip predictions to avoid log(0) errors, though sklearn handles this usually
    # Using sklearn's log_loss
    metric_score = log_loss(all_labels, all_preds, labels=[0, 1])

    return epoch_loss, metric_score


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    model_name,
    fold,
    patience=None,
):
    """
    Runs the full training loop with validation and early stopping.
    Saves the best model based on Validation Log Loss.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        optimizer (Optimizer): Optimizer.
        scheduler (Scheduler): Learning rate scheduler.
        device (str): Device.
        epochs (int): Total epochs.
        model_name (str): Name identifier for checkpointing.
        fold (int): Current fold number.
        patience (int, optional): Early stopping patience. If None, runs all epochs.
    """
    best_log_loss = float("inf")
    best_model_path = get_checkpoint_path(model_name, fold)
    patience_counter = 0

    logger.info(f"Starting training for {model_name} - Fold {fold}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, val_log_loss = validate_one_epoch(model, val_loader, device)

        # Step the scheduler (CosineAnnealing usually steps per epoch)
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics with full precision
        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val Log Loss: {val_log_loss} - "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint logic
        if val_log_loss < best_log_loss:
            best_log_loss = val_log_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience is not None and patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    logger.info(f"Training complete. Best Val Log Loss: {best_log_loss}")

    # Load best weights before returning
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model


def predict(model, dataloader, device, use_tta=True):
    """
    Generates predictions for the test set.
    Implements Test Time Augmentation (Horizontal Flip) if requested.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Test dataloader.
        device (str): Device.
        use_tta (bool): Whether to use Test Time Augmentation.

    Returns:
        tuple: (ids, probabilities)
    """
    model.eval()
    all_ids = []
    all_probs = []

    with torch.no_grad():
        for images, ids in dataloader:
            images = images.to(device)

            # Forward pass 1: Original
            logits = model(images)
            probs = torch.sigmoid(logits)

            if use_tta:
                # Forward pass 2: Horizontal Flip
                images_flipped = torch.flip(images, dims=[3])
                logits_flipped = model(images_flipped)
                probs_flipped = torch.sigmoid(logits_flipped)

                # Average probabilities
                probs = (probs + probs_flipped) / 2.0

            all_probs.append(probs.cpu().numpy())
            all_ids.extend(ids.numpy())

    all_probs = np.concatenate(all_probs).flatten()
    all_ids = np.array(all_ids)

    return all_ids, all_probs
