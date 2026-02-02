import os
import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config, seed_everything
from library.utils import (
    AverageMeter,
    get_logger,
    save_checkpoint,
    print_metrics,
)
from library.dataset import CervicalSpineDataset
from library.model import AnatomicallyAwareModel
from library.loss import WeightedFractureLoss

# Initialize logger
logger = get_logger(name="Train")


def train_one_epoch(
    model, loader, optimizer, loss_fn, device, epoch, scaler, scheduler=None
):
    """
    Trains the model for one epoch using Gradient Accumulation and Mixed Precision.
    """
    model.train()
    loss_meter = AverageMeter("Train Loss")

    # Zero gradients at the start of the epoch
    optimizer.zero_grad()

    num_batches = len(loader)

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Mixed Precision Forward Pass
        with autocast():
            logits = model(images)
            loss = loss_fn(logits, targets)

            # Normalize loss for gradient accumulation
            loss = loss / Config.GRAD_ACCUM_STEPS

        # Backward Pass (Scaled)
        scaler.scale(loss).backward()

        # Update weights only after accumulating enough gradients
        if (batch_idx + 1) % Config.GRAD_ACCUM_STEPS == 0 or (
            batch_idx + 1
        ) == num_batches:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Step optimizer and update scaler
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next accumulation cycle
            optimizer.zero_grad()

            # Step scheduler if it's per-step (optional, here we assume per-epoch usually,
            # but if OneCycleLR was used it would be here. We use CosineAnnealing per epoch below)

        # Update metrics (multiply back by accumulation steps to log "real" loss)
        loss_meter.update(loss.item() * Config.GRAD_ACCUM_STEPS, images.size(0))

    return loss_meter.avg


def validate(model, loader, loss_fn, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter("Val Loss")

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Optional: Use autocast for validation speedup
            with autocast():
                logits = model(images)
                loss = loss_fn(logits, targets)

            loss_meter.update(loss.item(), images.size(0))

    return loss_meter.avg


def run_training():
    """
    Main training loop with Early Stopping.
    """
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    logger.info(f"Starting training on device: {device}")

    # --- Data Loading ---
    logger.info("Initializing Datasets...")
    train_dataset = CervicalSpineDataset(
        metadata_path=Config.TRAIN_METADATA_PATH, phase="train", load_cached_data=True
    )
    val_dataset = CervicalSpineDataset(
        metadata_path=Config.VAL_METADATA_PATH, phase="val", load_cached_data=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Important for consistent batch sizes in accumulation
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- Model Setup ---
    logger.info("Initializing Model...")
    model = AnatomicallyAwareModel()
    model = model.to(device)

    # --- Optimization ---
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    loss_fn = WeightedFractureLoss().to(device)
    scaler = GradScaler()

    # --- Training Loop ---
    best_loss = float("inf")
    patience_counter = 0

    logger.info("Starting Training Loop...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, scaler
        )

        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)

        # Update Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Log Metrics
        metrics = {
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "LR": optimizer.param_groups[0]["lr"],
            "Time": f"{elapsed:.2f}s",
        }
        print_metrics(metrics, logger)

        # --- Early Stopping & Checkpointing ---
        # Check if this is the best model
        is_best = val_loss < (best_loss - Config.MIN_DELTA)

        if is_best:
            best_loss = val_loss
            patience_counter = 0
            logger.info(
                f"New best model found (Loss: {best_loss}). Saving checkpoint..."
            )

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": best_loss,
                },
                is_best=True,
                filepath=Config.CHECKPOINT_PATH,
            )
        else:
            patience_counter += 1
            logger.info(
                f"Validation loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            logger.info("Early stopping triggered. Training finished.")
            break

        # Cleanup to prevent memory leaks
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Training complete. Best Validation Loss: {best_loss}")
