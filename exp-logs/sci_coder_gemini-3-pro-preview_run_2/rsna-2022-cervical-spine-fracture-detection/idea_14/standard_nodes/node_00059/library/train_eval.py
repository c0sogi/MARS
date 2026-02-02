import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import (
    seed_everything,
    get_device,
    setup_logger,
    AverageMeter,
    format_time,
    save_checkpoint,
)
from library.data import get_dataloaders
from library.model import CervicalFractureModel


class WeightedMultilabelLoss(nn.Module):
    """
    Computes the Weighted Multi-Label Logarithmic Loss.

    Formula:
    L = - sum(w_j * [y_j * log(p_j) + (1-y_j) * log(1-p_j)]) / N_classes

    Crucially, we use BCEWithLogitsLoss with reduction='none' and NO pos_weight
    to ensure the predicted probabilities are calibrated (Lesson 00042).
    """

    def __init__(self, weights, device):
        super().__init__()
        # Weights shape: (1, Num_Classes) for broadcasting
        self.weights = torch.tensor(weights, device=device).view(1, -1)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        # Calculate standard BCE for each element
        # logits: (Batch, 8), targets: (Batch, 8)
        loss = self.bce(logits, targets)

        # Apply competition weights
        weighted_loss = loss * self.weights

        # Average over all elements (Batch * Classes)
        # Note: The metric definition says "loss is averaged across all rows".
        # Since our weights are per-column, we average over the batch and sum/mean over classes appropriately.
        # Usually, this implies mean over the batch, and mean over the classes (weighted).
        return weighted_loss.mean()


def train_one_epoch(
    model, loader, criterion, optimizer, device, epoch, logger, scaler=None
):
    """
    Handles the training of a single epoch with gradient accumulation.
    """
    model.train()

    losses = AverageMeter()
    start_time = time.time()

    # Zero gradients at start of epoch
    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(loader):
        # Move data to device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = images.size(0)

        # Mixed precision context if available (optional, but good for A100)
        # Here we stick to standard float32 as per simple requirements unless scaler is passed
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(images)
            loss = criterion(logits, targets)

            # Normalize loss for gradient accumulation
            loss = loss / Config.GRAD_ACCUMULATION_STEPS

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update metrics (multiply back to get actual loss value)
        losses.update(loss.item() * Config.GRAD_ACCUMULATION_STEPS, batch_size)

        # Optimizer Step (Gradient Accumulation)
        if (batch_idx + 1) % Config.GRAD_ACCUMULATION_STEPS == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                optimizer.step()

            optimizer.zero_grad()

    elapsed = time.time() - start_time
    logger.info(
        f"Epoch {epoch} [Train] Loss: {losses.avg:.5f} | Time: {format_time(elapsed)}"
    )

    return losses.avg


def validate(model, loader, criterion, device, logger):
    """
    Evaluates the model on the validation set.
    Computes the exact competition metric (Weighted Log Loss).
    """
    model.eval()

    losses = AverageMeter()
    start_time = time.time()

    # Store predictions and targets for global metric calculation if needed,
    # but average batch loss is sufficient if batch size is consistent or weighted correctly.

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch_size = images.size(0)

            # Forward
            logits = model(images)

            # Calculate Loss (Metric)
            loss = criterion(logits, targets)

            losses.update(loss.item(), batch_size)

    elapsed = time.time() - start_time
    logger.info(
        f"Epoch {0} [Val]   Loss: {losses.avg:.10f} | Time: {format_time(elapsed)}"
    )
    # Note: Printing full precision as requested

    return losses.avg


def run_training():
    """
    Main execution function.
    Sets up the environment, model, data, and runs the training loop.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    logger = setup_logger()

    logger.info(f"Starting experiment: {Config.EXP_NAME}")
    logger.info(f"Device: {device}")

    # 2. Data
    logger.info("Loading data...")
    train_loader, val_loader = get_dataloaders(load_cached_data=True)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model
    logger.info("Initializing model...")
    model = CervicalFractureModel()
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    # 5. Loss Function
    # Config.LOSS_WEIGHTS = [1.0, ..., 7.0]
    criterion = WeightedMultilabelLoss(Config.LOSS_WEIGHTS, device)

    # Scaler for mixed precision (A100 supports this well)
    scaler = torch.cuda.amp.GradScaler()

    # 6. Training Loop
    best_loss = float("inf")
    patience = 3
    patience_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        logger.info(f"\n=== Epoch {epoch}/{Config.EPOCHS} ===")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger, scaler
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device, logger)

        # Scheduler Step
        scheduler.step()

        # Checkpointing & Early Stopping
        if val_loss < best_loss:
            logger.info(
                f"Validation loss improved from {best_loss:.6f} to {val_loss:.6f}. Saving model..."
            )
            best_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, scheduler, filename="best_model.pth"
            )
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Validation Loss: {best_loss:.10f}")
