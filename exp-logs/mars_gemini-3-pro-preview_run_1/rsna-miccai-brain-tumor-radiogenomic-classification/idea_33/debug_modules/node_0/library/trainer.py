import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from library.config import (
    DEVICE,
    WORKING_DIR,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEED,
)
from library.utils import (
    AverageMeter,
    calculate_roc_auc,
    save_checkpoint,
    get_logger,
    seed_everything,
)
from library.data_processing import get_dataloaders
from library.model import RARVEfficientNet


def train_one_epoch(train_loader, model, criterion, optimizer, epoch, logger, device):
    """
    Executes one training epoch.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()

    # Switch to train mode
    model.train()

    start = time.time()

    all_targets = []
    all_scores = []

    for i, (images, targets, _) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device).view(-1, 1)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        losses.update(loss.item(), images.size(0))

        # Collect targets and scores for AUC calculation
        # Apply sigmoid to logits for scores
        scores = torch.sigmoid(outputs).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        all_targets.extend(targets_np)
        all_scores.extend(scores)

        # Measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

    # Calculate epoch metrics
    epoch_auc = calculate_roc_auc(all_targets, all_scores)

    logger.info(
        f"Epoch: [{epoch + 1}] "
        f"Train Loss: {losses.avg:.10f} "
        f"Train AUC: {epoch_auc:.10f}"
    )

    return losses.avg, epoch_auc


def validate(val_loader, model, criterion, logger, device):
    """
    Evaluates the model on the validation set.
    """
    losses = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    all_targets = []
    all_scores = []

    with torch.no_grad():
        for i, (images, targets, _) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device).view(-1, 1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Record loss
            losses.update(loss.item(), images.size(0))

            # Collect scores
            scores = torch.sigmoid(outputs).cpu().numpy()
            targets_np = targets.cpu().numpy()

            all_targets.extend(targets_np)
            all_scores.extend(scores)

    # Calculate metrics
    val_auc = calculate_roc_auc(all_targets, all_scores)

    logger.info(
        f"Validation Loss: {losses.avg:.10f} " f"Validation AUC: {val_auc:.10f}"
    )

    return losses.avg, val_auc


def run_training(load_cached_data=True, max_epochs=EPOCHS, patience=5):
    """
    Orchestrates the training process.
    """
    # 1. Setup
    seed_everything(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)
    logger = get_logger("trainer")

    logger.info(f"Starting training on device: {DEVICE}")
    logger.info(
        f"Hyperparameters: LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, BS={max_epochs}"
    )

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    model = RARVEfficientNet()
    model = model.to(DEVICE)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(max_epochs):
        logger.info(f"--- Epoch {epoch + 1}/{max_epochs} ---")

        # Train
        train_loss, train_auc = train_one_epoch(
            train_loader, model, criterion, optimizer, epoch, logger, DEVICE
        )

        # Validate
        val_loss, val_auc = validate(val_loader, model, criterion, logger, DEVICE)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Current Learning Rate: {current_lr:.8f}")

        # Checkpointing & Early Stopping
        is_best = val_auc > best_auc
        if is_best:
            best_auc = val_auc
            patience_counter = 0
            logger.info(f"New Best AUC found: {best_auc:.10f}. Saving model...")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_auc": best_auc,
                "optimizer": optimizer.state_dict(),
            },
            filename=f"checkpoint_epoch_{epoch+1}.pth",
            is_best=is_best,
        )

        # Early Stopping Trigger
        if patience_counter >= patience:
            logger.info("Early stopping triggered. Training finished.")
            break

    logger.info(f"Training complete. Best Validation AUC: {best_auc:.10f}")
