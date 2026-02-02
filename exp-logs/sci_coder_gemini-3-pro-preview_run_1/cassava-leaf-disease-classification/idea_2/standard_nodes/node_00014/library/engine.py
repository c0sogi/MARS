import os
import time
import torch
import torch.nn as nn
import numpy as np
from library.config import CFG
from library.utils import AverageMeter, get_score, print_metrics
from library.loss import SoftTargetCrossEntropy


def train_one_epoch(
    epoch, model, train_loader, optimizer, device, scheduler=None, logger=None
):
    """
    Trains the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): Optimizer instance.
        device (torch.device): Device to compute on.
        scheduler (lr_scheduler, optional): Learning rate scheduler.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()

    # Loss function for soft targets (MixUp/CutMix)
    criterion = SoftTargetCrossEntropy()

    loss_meter = AverageMeter()

    start_time = time.time()

    for step, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        y_preds = model(images)

        # Compute loss
        loss = criterion(y_preds, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Update loss meter
        loss_meter.update(loss.item(), batch_size)

        if scheduler is not None and isinstance(
            scheduler, torch.optim.lr_scheduler.OneCycleLR
        ):
            scheduler.step()

        if CFG.debug and step > 10:
            break

    elapsed = time.time() - start_time

    if logger:
        logger.info(
            f"Epoch {epoch} - Train Loss: {loss_meter.avg} - Time: {elapsed:.2f}s"
        )

    return loss_meter.avg


def validate(model, val_loader, device, logger=None):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to compute on.
        logger (logging.Logger, optional): Logger instance.

    Returns:
        dict: Dictionary containing 'loss' and 'accuracy'.
    """
    model.eval()

    # Standard CrossEntropy for validation (hard targets)
    criterion = nn.CrossEntropyLoss()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    preds_list = []
    targets_list = []

    start_time = time.time()

    with torch.no_grad():
        for step, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            # Forward pass
            y_preds = model(images)

            # Compute loss
            loss = criterion(y_preds, targets)

            # Update meters
            loss_meter.update(loss.item(), batch_size)

            # Store predictions and targets for accuracy calculation
            preds_list.append(y_preds.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

            if CFG.debug and step > 10:
                break

    # Concatenate all batches
    preds_arr = np.concatenate(preds_list)
    targets_arr = np.concatenate(targets_list)

    # Calculate accuracy
    accuracy = get_score(targets_arr, preds_arr)

    elapsed = time.time() - start_time

    metrics = {"loss": loss_meter.avg, "accuracy": accuracy}

    if logger:
        logger.info(
            f"Validation - Loss: {metrics['loss']} - Accuracy: {metrics['accuracy']} - Time: {elapsed:.2f}s"
        )

    return metrics


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    logger,
    patience=5,
):
    """
    Main training loop with Early Stopping.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        scheduler (lr_scheduler): Learning rate scheduler.
        device (torch.device): Device.
        epochs (int): Total number of epochs.
        logger (logging.Logger): Logger.
        patience (int): Number of epochs to wait for improvement before stopping.
    """
    best_acc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(CFG.output_dir, CFG.model_save_name)

    # Ensure output directory exists
    os.makedirs(CFG.output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        logger.info(f"\n{'='*20} Epoch {epoch} / {epochs} {'='*20}")

        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, device, scheduler, logger
        )

        # Validate
        val_metrics = validate(model, val_loader, device, logger)
        val_acc = val_metrics["accuracy"]

        # Step scheduler (CosineAnnealingLR steps per epoch)
        if scheduler is not None and not isinstance(
            scheduler, torch.optim.lr_scheduler.OneCycleLR
        ):
            scheduler.step()

        # Early Stopping and Checkpointing
        if val_acc > best_acc:
            logger.info(f"Validation Accuracy Improved ({best_acc} ---> {val_acc})")
            best_acc = val_acc
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Model Saved to {best_model_path}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement in Accuracy. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training Complete. Best Validation Accuracy: {best_acc}")
