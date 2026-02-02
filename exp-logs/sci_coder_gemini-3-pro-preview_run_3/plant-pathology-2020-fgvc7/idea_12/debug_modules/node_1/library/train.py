import os
import time
import torch
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from library.config import Config
from library.utils import get_logger, compute_metric

# Initialize logger
logger = get_logger(__name__)


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, model_ema=None):
    """
    Trains the model for one epoch using AMP and Deep Supervision.

    Args:
        model: The PyTorch model to train.
        loader: DataLoader for the training set.
        optimizer: The optimizer.
        loss_fn: The loss function (DeepSupervisionLoss).
        device: The device to train on.
        scaler: GradScaler for AMP.
        model_ema: ModelEMA instance (optional).

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for _, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Automatic Mixed Precision Context
        with autocast(enabled=Config.use_amp):
            # Model returns tuple (p3, p4, p5) during training for Deep Supervision
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update EMA model
        if model_ema is not None:
            model_ema.update(model)

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, loss_fn, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model to evaluate (typically the EMA model).
        loader: DataLoader for the validation set.
        loss_fn: The loss function.
        device: The device to evaluate on.

    Returns:
        tuple: (average_loss, roc_auc_score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            # In eval mode, model returns only the main head logits (p3)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply Softmax to get probabilities for mutually exclusive classes
            # (healthy, multiple_diseases, rust, scab)
            probs = torch.softmax(outputs, dim=1)

            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(preds, axis=0)
    all_targets = np.concatenate(targets, axis=0)

    # Compute Mean Column-wise ROC AUC
    metric = compute_metric(all_targets, all_preds)

    return epoch_loss, metric


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    num_epochs,
    patience,
    save_path,
    model_ema=None,
):
    """
    Orchestrates the training process including logging, early stopping, and saving.

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        loss_fn: Loss function.
        device: Device.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience.
        save_path: Path to save the best model.
        model_ema: ModelEMA instance.

    Returns:
        float: The best validation metric achieved.
    """
    scaler = GradScaler(enabled=Config.use_amp)
    best_metric = -float("inf")
    best_epoch = 0
    early_stop_counter = 0

    logger.info(f"Starting training for {num_epochs} epochs on device {device}")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train Step
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, model_ema
        )

        # Validation Step
        # Use EMA model for validation if available, otherwise standard model
        val_model = model_ema.module if model_ema else model
        val_loss, val_metric = validate(val_model, val_loader, loss_fn, device)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - start_time

        # Log metrics with full precision
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Time: {elapsed:.2f}s - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_metric}"
        )

        # Save Best Model
        if val_metric > best_metric:
            best_metric = val_metric
            best_epoch = epoch
            early_stop_counter = 0

            logger.info(f"New best metric: {best_metric}. Saving model to {save_path}")
            torch.save(val_model.state_dict(), save_path)
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= patience:
            logger.info(
                f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with AUC {best_metric}"
            )
            break

    return best_metric
