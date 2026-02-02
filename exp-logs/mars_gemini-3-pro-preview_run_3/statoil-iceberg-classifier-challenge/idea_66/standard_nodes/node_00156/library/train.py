import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library import utils, model, data_loader


def train_one_epoch(
    train_loader, model_instance, criterion, optimizer, device, epoch, logger
):
    """
    Trains the model for one epoch using Interaction-Aware Multi-Sample Dropout.
    """
    model_instance.train()
    losses = utils.AverageMeter()

    for i, (images, angles, targets) in enumerate(train_loader):
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device)

        # Forward pass
        # In training mode, returns (Batch, Num_Samples)
        logits = model_instance(images, angles)

        # Prepare targets for multi-sample loss
        # Expand targets from (Batch) to (Batch, Num_Samples)
        # targets is (Batch), view as (Batch, 1), expand to (Batch, Num_Samples)
        targets_expanded = targets.view(-1, 1).expand_as(logits)

        # Compute loss
        loss = criterion(logits, targets_expanded)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(val_loader, model_instance, criterion, device, logger):
    """
    Evaluates the model on the validation set.
    Uses the averaged logits (inference mode) for calculation.
    """
    model_instance.eval()
    losses = utils.AverageMeter()

    with torch.no_grad():
        for i, (images, angles, targets) in enumerate(val_loader):
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device)

            # Forward pass
            # In eval mode, returns (Batch, 1) - the average of branches
            logits = model_instance(images, angles)

            # Targets need to match shape (Batch, 1)
            targets_view = targets.view(-1, 1)

            loss = criterion(logits, targets_view)

            losses.update(loss.item(), images.size(0))

    return losses.avg


def run_fold(fold_index):
    """
    Runs the training pipeline for a specific fold.
    """
    # Setup
    utils.set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create working directory for logs/checkpoints if not exists
    Config.setup_directories()

    # Logger
    log_file = os.path.join(Config.WORKING_DIR, f"train_fold_{fold_index}.log")
    logger = utils.setup_logger(log_file, name=f"fold_{fold_index}")

    logger.info(f"Starting training for Fold {fold_index}")
    logger.info(f"Device: {device}")

    # Data Loaders
    train_loader, val_loader = data_loader.get_train_val_loaders(
        fold_index, load_cached_data=True
    )

    # Model
    net = model.IAMSI_CNN().to(device)

    # Optimizer (AdamW with constant LR)
    optimizer = optim.AdamW(
        net.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            train_loader, net, criterion, optimizer, device, epoch, logger
        )

        # Validate
        val_loss = validate(val_loader, net, criterion, device, logger)

        elapsed = time.time() - start_time

        # Logging with full precision
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Time: {elapsed:.2f}s - "
            f"Train Loss: {train_loss:.10f} - "
            f"Val Loss: {val_loss:.10f}"
        )

        # Checkpointing & Early Stopping
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
            logger.info(f"New best model found! Val Loss: {val_loss:.10f}")
        else:
            patience_counter += 1

        # Save checkpoint
        utils.save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss,
                "fold": fold_index,
            },
            is_best,
            fold_index,
        )

        if patience_counter >= Config.PATIENCE:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break

    logger.info(f"Fold {fold_index} finished. Best Val Loss: {best_loss:.10f}")
    return best_loss
