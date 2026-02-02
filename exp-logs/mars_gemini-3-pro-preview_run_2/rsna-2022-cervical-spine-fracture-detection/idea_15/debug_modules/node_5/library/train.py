import os
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch import autocast

from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    MetricMonitor,
    save_checkpoint,
)
from library.loss import WeightedMultiLabelLogLoss
from library.data import get_dataloaders
from library.model import CalibratedSequenceNetwork


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, scaler):
    """
    Trains the model for one epoch using gradient accumulation and mixed precision.
    """
    model.train()
    metric_monitor = MetricMonitor()

    # Reset gradients at the start of the epoch
    optimizer.zero_grad()

    num_batches = len(train_loader)

    for batch_idx, (images, targets) in enumerate(train_loader):
        # Move data to device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = images.size(0)

        # Mixed Precision Forward Pass
        with autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(images)
            loss = criterion(logits, targets)
            # Scale loss for gradient accumulation
            loss = loss / Config.ACCUMULATION_STEPS

        # Backward Pass with Scaler
        scaler.scale(loss).backward()

        # Update weights every ACCUMULATION_STEPS or at the end of the epoch
        if ((batch_idx + 1) % Config.ACCUMULATION_STEPS == 0) or (
            (batch_idx + 1) == num_batches
        ):
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients
            optimizer.zero_grad()

        # Update metrics (multiply loss back by accumulation steps for reporting)
        loss_value = loss.item() * Config.ACCUMULATION_STEPS
        metric_monitor.update(loss_value, batch_size)

    return metric_monitor.avg


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    metric_monitor = MetricMonitor()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch_size = images.size(0)

            # Forward pass (Mixed precision is optional for val, but good for consistency)
            with autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(images)
                loss = criterion(logits, targets)

            metric_monitor.update(loss.item(), batch_size)

    return metric_monitor.avg


def run_training(debug=Config.DEBUG):
    """
    Main function to orchestrate the training process.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    log_file = os.path.join(Config.WORKING_DIR, "train.log")
    logger = get_logger("training", log_file)

    device = Config.DEVICE
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    logger.info(f"Loading data (Debug={debug})...")
    train_loader, val_loader = get_dataloaders(load_cached_data=True, debug=debug)

    # 3. Model Initialization
    logger.info(f"Initializing model: {Config.BACKBONE_NAME}...")
    model = CalibratedSequenceNetwork()
    model.to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    criterion = WeightedMultiLabelLogLoss()
    scaler = GradScaler(enabled=(device == "cuda"))

    # 5. Training Loop
    best_loss = float("inf")
    patience_counter = 0

    logger.info("Starting training...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, scaler
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        curr_lr = optimizer.param_groups[0]["lr"]

        # Logging
        logger.info(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {curr_lr:.2e} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss}"
        )

        # Checkpoint & Early Stopping
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
            logger.info(f"New best validation loss: {best_loss}")
        else:
            patience_counter += 1
            logger.info(
                f"Early stopping counter: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        save_state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_loss": best_loss,
        }

        save_checkpoint(
            save_state, is_best, Config.WORKING_DIR, best_model_name="best_model.pth"
        )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered. Training finished.")
            break

        # Cleanup to prevent OOM
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Training complete. Best Validation Loss: {best_loss}")
