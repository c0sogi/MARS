import os
import time
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import set_seed, setup_logger
from library.data import get_fold_loaders
from library.model import CCTICNN


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for batch_idx, ((images, angles), targets, _) in enumerate(loader):
        # Move data to device
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape is (B, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item() * images.size(0)
        num_samples += images.size(0)

    avg_loss = running_loss / num_samples
    return avg_loss


def validate(model, loader, criterion, device, logger):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, ((images, angles), targets, _) in enumerate(loader):
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

    avg_loss = running_loss / num_samples
    return avg_loss


def run_fold(fold_index):
    """
    Runs the training pipeline for a specific fold.
    Handles initialization, training loop, early stopping, and checkpointing.
    """
    # Initialize directory for logs if not exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Setup Logger
    log_file = os.path.join(Config.WORKING_DIR, f"train_fold_{fold_index}.log")
    logger = setup_logger(f"Fold_{fold_index}", log_file)

    logger.info(f"Starting training for Fold {fold_index}")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Device configuration
    device = torch.device(Config.DEVICE)

    # Data Loaders
    # get_fold_loaders handles the caching and splitting logic internally
    train_loader, val_loader = get_fold_loaders(fold_index, load_cached_data=True)

    # Initialize Model
    model = CCTICNN()
    model.to(device)

    # Optimizer: AdamW with constant LR
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Loss Function: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    # Early Stopping Variables
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(
        Config.CHECKPOINT_DIR, f"model_best_fold_{fold_index}.pth"
    )

    # Training Loop
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device, logger)

        duration = time.time() - start_time

        # Log full precision metrics
        logger.info(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Time: {duration}s"
        )

        # Checkpoint & Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"Validation loss improved. Saved best model to {best_model_path}"
            )
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    # Save final checkpoint (state of the last epoch)
    final_ckpt_path = os.path.join(
        Config.CHECKPOINT_DIR, f"checkpoint_fold_{fold_index}.pth"
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
        },
        final_ckpt_path,
    )

    logger.info(f"Fold {fold_index} finished. Best Validation Loss: {best_val_loss}")

    return best_val_loss
