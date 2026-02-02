import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, get_logger, AverageMeter, kl_divergence
from library.data import get_dataloader, mixup_data
from library.model import AttentiveDualScaleNetwork


def train_one_epoch(epoch, model, loader, optimizer, criterion, device, logger):
    """
    Handles the training of one epoch with MixUp augmentation.
    """
    model.train()
    losses = AverageMeter()

    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Unpack dual-stream inputs
        x_eeg, x_spec = inputs
        x_eeg = x_eeg.to(device, non_blocking=True)
        x_spec = x_spec.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply MixUp
        # We mix both streams with the same lambda to preserve consistency
        mixed_x_eeg, mixed_x_spec, y_a, y_b, lam = mixup_data(
            x_eeg, x_spec, targets, alpha=1.0, device=device
        )

        optimizer.zero_grad()

        # Forward pass
        logits = model((mixed_x_eeg, mixed_x_spec))

        # KLDivLoss expects log-probabilities
        log_probs = F.log_softmax(logits, dim=1)

        # MixUp Loss
        loss = lam * criterion(log_probs, y_a) + (1 - lam) * criterion(log_probs, y_b)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        losses.update(loss.item(), x_eeg.size(0))

    elapsed = time.time() - start_time
    logger.info(f"Epoch {epoch} [Train] Loss: {losses.avg} | Time: {elapsed:.2f}s")

    return losses.avg


def validate(model, loader, criterion, device, logger):
    """
    Evaluates the model on the validation set.
    Computes both the Loss and the specific KL Divergence metric.
    """
    model.eval()
    losses = AverageMeter()
    metric_meter = AverageMeter()

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            x_eeg, x_spec = inputs
            x_eeg = x_eeg.to(device, non_blocking=True)
            x_spec = x_spec.to(device, non_blocking=True)
            targets_dev = targets.to(device, non_blocking=True)

            # Forward pass
            logits = model((x_eeg, x_spec))

            # Loss calculation (Log Softmax for KLDivLoss)
            log_probs = F.log_softmax(logits, dim=1)
            loss = criterion(log_probs, targets_dev)
            losses.update(loss.item(), x_eeg.size(0))

            # Metric calculation (Probabilities for custom KL metric)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            targets_np = targets.numpy()

            score = kl_divergence(targets_np, probs)
            metric_meter.update(score, x_eeg.size(0))

    elapsed = time.time() - start_time
    logger.info(
        f"Epoch - [Val] Loss: {losses.avg} | Metric (KL): {metric_meter.avg} | Time: {elapsed:.2f}s"
    )

    return losses.avg, metric_meter.avg


def run_training(debug=False, load_cached_data=True):
    """
    Main execution function for training the Attentive Dual-Scale Fusion Network.

    Args:
        debug (bool): If True, runs a shorter version for debugging.
        load_cached_data (bool): If True, attempts to load pre-processed .npy files.
    """
    # Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    log_file = os.path.join(Config.WORKING_DIR, "training.log")
    logger = get_logger(log_file)

    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # Data Loaders
    logger.info("Initializing DataLoaders...")
    train_loader = get_dataloader(
        mode="train",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=load_cached_data,
        shuffle=True,
    )
    val_loader = get_dataloader(
        mode="val",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=load_cached_data,
        shuffle=False,
    )

    # Model
    logger.info("Initializing Model...")
    model = AttentiveDualScaleNetwork()
    model.to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    num_epochs = 2 if debug else Config.EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    # Loss Function
    # KLDivLoss with batchmean is standard for KL divergence optimization
    criterion = nn.KLDivLoss(reduction="batchmean")

    # Training Loop
    best_metric = float("inf")
    patience_counter = 0

    logger.info(f"Starting training for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")

        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, optimizer, criterion, device, logger
        )

        # Validate
        val_loss, val_metric = validate(model, val_loader, criterion, device, logger)

        # Scheduler Step
        scheduler.step()

        # Checkpointing & Early Stopping
        # We optimize for the competition metric (KL Divergence)
        if val_metric < best_metric:
            logger.info(
                f"Metric improved from {best_metric} to {val_metric}. Saving model..."
            )
            best_metric = val_metric
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"Metric did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Validation Metric: {best_metric}")
