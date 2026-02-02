import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import setup_logger, AverageMeter, set_seed
from library.model import DPSCACNN
from library.data_loader import get_loaders


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, logger):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    start_time = time.time()

    for batch_idx, (images, angles, targets) in enumerate(loader):
        # Move data to device
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).view(-1, 1)  # Ensure shape is (B, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

    elapsed = time.time() - start_time
    logger.info(f"Epoch {epoch} [Train] Loss: {losses.avg} Time: {elapsed:.2f}s")

    return losses.avg


def validate(model, loader, criterion, device, logger):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()

    start_time = time.time()

    with torch.no_grad():
        for images, angles, targets in loader:
            # Move data to device
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).view(-1, 1)

            # Forward pass
            outputs = model(images, angles)

            # Compute loss
            loss = criterion(outputs, targets)

            # Update metrics
            losses.update(loss.item(), images.size(0))

    elapsed = time.time() - start_time
    logger.info(f"Epoch Val [Validate] Loss: {losses.avg} Time: {elapsed:.2f}s")

    return losses.avg


def run_fold(fold, load_cached_data=True):
    """
    Runs training and validation for a specific fold.

    Args:
        fold (int): The fold index.
        load_cached_data (bool): Whether to use cached data.
    """
    # Setup logging
    log_file = os.path.join(Config.WORKING_DIR, f"train_fold_{fold}.log")
    logger = setup_logger(f"Fold_{fold}", log_file)
    logger.info(f"Starting Fold {fold}")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Device
    device = torch.device(Config.DEVICE)
    logger.info(f"Device: {device}")

    # Data Loaders
    logger.info("Loading data...")
    train_loader, val_loader, _ = get_loaders(
        fold=fold, load_cached_data=load_cached_data
    )

    # Model Initialization
    logger.info("Initializing DPSCA-CNN model...")
    model = DPSCACNN().to(device)

    # Optimizer (AdamW)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function (BCEWithLogitsLoss)
    # Note: Targets are 0 or 1, logits are raw scores.
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop Variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, f"model_fold_{fold}.pth")

    logger.info(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, logger
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device, logger)

        logger.info(f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}")

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"New best model saved to {best_model_path} (Val Loss: {val_loss})"
            )
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Fold {fold} finished. Best Val Loss: {best_val_loss}")
    return best_val_loss
