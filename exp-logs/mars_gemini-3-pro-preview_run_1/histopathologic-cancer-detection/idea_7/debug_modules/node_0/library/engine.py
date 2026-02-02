import os
import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import MetricMonitor, setup_logger
from library.network import DenseNet121GeM


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, logger):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.
        logger (logging.Logger): Logger instance.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    # Label smoothing factor
    smoothing = Config.LABEL_SMOOTHING

    for batch_idx, data in enumerate(train_loader):
        images = data["image"].to(device, dtype=torch.float)
        targets = data["target"].to(device, dtype=torch.float)

        # Apply Label Smoothing manually for BCE
        # y_smooth = y * (1 - alpha) + 0.5 * alpha
        smoothed_targets = targets * (1 - smoothing) + 0.5 * smoothing

        optimizer.zero_grad()

        # Forward pass
        # Model outputs logits of shape (Batch, 1)
        outputs = model(images).view(-1)

        loss = criterion(outputs, smoothed_targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        metric_monitor.update(loss.item(), images.size(0))

    return metric_monitor.avg


def evaluate(model, val_loader, criterion, device, logger):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run evaluation on.
        logger (logging.Logger): Logger instance.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    metric_monitor = MetricMonitor()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data in val_loader:
            images = data["image"].to(device, dtype=torch.float)
            targets = data["target"].to(device, dtype=torch.float)

            outputs = model(images).view(-1)
            loss = criterion(outputs, targets)  # No smoothing during validation

            metric_monitor.update(loss.item(), images.size(0))

            # Apply sigmoid to convert logits to probabilities for AUC calculation
            probs = torch.sigmoid(outputs)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(probs.cpu().numpy())

    avg_loss = metric_monitor.avg

    # Handle edge case where only one class is present in the batch/loader (though unlikely in valid set)
    try:
        auc = roc_auc_score(all_targets, all_predictions)
    except ValueError:
        auc = 0.5
        logger.warning(
            "ROC AUC calculation failed (likely only one class present). Defaulting to 0.5."
        )

    return avg_loss, auc


def fit(fold, train_loader, val_loader, device=None):
    """
    Orchestrates the training process for a single fold.

    Args:
        fold (int): The current fold index.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        device (torch.device, optional): Device to run on. Defaults to Config.DEVICE.
    """
    if device is None:
        device = torch.device(Config.DEVICE)

    # Create directory for saving models if it doesn't exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Setup Logger
    log_file = os.path.join(Config.WORKING_DIR, f"train_fold{fold}.log")
    logger = setup_logger(f"Fold{fold}", log_file)

    logger.info(f"Starting training for Fold {fold}")
    logger.info(f"Device: {device}")

    # Initialize Model
    model = DenseNet121GeM(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # Loss Function (BCEWithLogitsLoss is numerically stable)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(
        Config.WORKING_DIR, f"{Config.MODEL_NAME}_fold{fold}_best.pth"
    )

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )

        # Validate
        val_loss, val_auc = evaluate(model, val_loader, criterion, device, logger)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Logging (Full precision for metrics as requested)
        logger.info(
            f"Epoch {epoch}/{Config.EPOCHS} - "
            f"Time: {elapsed:.2f}s - "
            f"LR: {current_lr:.2e} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val AUC: {val_auc}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model found! Saved to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training finished for Fold {fold}. Best AUC: {best_auc}")

    # Clean up to free memory
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
