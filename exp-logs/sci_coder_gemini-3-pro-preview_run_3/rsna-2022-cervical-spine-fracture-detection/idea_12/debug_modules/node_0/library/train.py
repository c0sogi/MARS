import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import get_dataloaders
from library.model import RSNAModel
from library.loss import ImplicitWeightedLoss


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, epoch, logger):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Context
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, targets)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / num_batches
    elapsed = time.time() - start_time

    logger.info(f"Epoch {epoch} | Train Loss: {avg_loss} | Time: {elapsed:.2f}s")

    return avg_loss


def validate(model, loader, criterion, device, logger):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    num_batches = len(loader)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)

            running_loss += loss.item()

    avg_loss = running_loss / num_batches
    logger.info(f"Validation Loss: {avg_loss}")

    return avg_loss


def run_training(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG, patience=5
):
    """
    Main training pipeline.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        debug (bool): If True, runs on a small subset of data.
        patience (int): Early stopping patience.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger("training")
    device = torch.device(Config.DEVICE)

    logger.info(f"Starting training on device: {device}")
    logger.info(
        f"Hyperparameters: Epochs={epochs}, Batch Size={batch_size}, Debug={debug}"
    )

    # 2. Data
    train_loader, val_loader = get_dataloaders(
        train_batch_size=batch_size, val_batch_size=batch_size * 2, debug=debug
    )

    # 3. Model
    model = RSNAModel(pretrained=True)
    model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Decoupled Cosine Annealing
    # T_max is set to 1.5x the number of epochs as requested
    t_max = int(epochs * Config.T_MAX_MULTIPLIER)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=t_max, eta_min=Config.MIN_LR
    )

    # Loss & Scaler
    criterion = ImplicitWeightedLoss()
    scaler = GradScaler()

    # 5. Training Loop
    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(1, epochs + 1):
        logger.info(f"--- Epoch {epoch}/{epochs} ---")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch, logger
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device, logger)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Current Learning Rate: {current_lr}")

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            logger.info(
                f"Validation loss improved from {best_val_loss} to {val_loss}. Saving model..."
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logger.info(
                f"Validation loss did not improve. Counter: {early_stop_counter}/{patience}"
            )

        if early_stop_counter >= patience:
            logger.info("Early stopping triggered. Training finished.")
            break

    logger.info(f"Training complete. Best Validation Loss: {best_val_loss}")
    logger.info(f"Best model saved to: {Config.MODEL_SAVE_PATH}")
