import os
import time
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    AverageMeter,
    save_checkpoint,
)
from library.loss import WeightedMultiLabelLoss
from library.dataset import RSNADataset, get_transforms, cache_image_paths
from library.model import CervicalSpineModel


def train_one_epoch(
    loader, model, criterion, optimizer, scaler, epoch, logger, accum_iter
):
    """
    Handles the training of one epoch with Gradient Accumulation and Mixed Precision.
    """
    batch_time = AverageMeter()
    losses = AverageMeter()

    # Switch to train mode
    model.train()

    start = time.time()
    optimizer.zero_grad()

    for i, (images, targets) in enumerate(loader):
        # Move data to device
        images = images.to(Config.device, non_blocking=True)
        targets = targets.to(Config.device, non_blocking=True)

        # Mixed Precision Forward Pass
        with autocast():
            logits = model(images)
            loss = criterion(logits, targets)
            # Normalize loss for gradient accumulation
            loss = loss / accum_iter

        # Backward Pass with Scaler
        scaler.scale(loss).backward()

        # Update weights every accum_iter steps
        if (i + 1) % accum_iter == 0:
            # Unscale to clip gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

            # Step optimizer
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Update metrics (multiply back by accum_iter to log the actual loss value)
        losses.update(loss.item() * accum_iter, images.size(0))
        batch_time.update(time.time() - start)
        start = time.time()

        # Logging
        if (i + 1) % Config.print_freq == 0:
            logger.info(
                f"Epoch: [{epoch + 1}][{i + 1}/{len(loader)}] "
                f"Batch Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f})"
            )

    return losses.avg


def validate(loader, model, criterion, logger):
    """
    Evaluates the model on the validation set.
    """
    losses = AverageMeter()

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        start = time.time()
        for i, (images, targets) in enumerate(loader):
            images = images.to(Config.device, non_blocking=True)
            targets = targets.to(Config.device, non_blocking=True)

            # Mixed precision inference (optional but faster)
            with autocast():
                logits = model(images)
                loss = criterion(logits, targets)

            losses.update(loss.item(), images.size(0))

    logger.info(f"Validation Loss: {losses.avg}")  # Full precision print
    return losses.avg


def run_training(
    debug=Config.debug,
    epochs=Config.epochs,
    load_cached_data=True,
):
    """
    Main execution function for the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.seed)
    logger = get_logger("train.log")
    logger.info("Starting training run...")
    logger.info(f"Device: {Config.device}")
    logger.info(f"Working Directory: {Config.working_dir}")

    # 2. Load Metadata
    train_df = pd.read_csv(Config.train_metadata_path)
    val_df = pd.read_csv(Config.val_metadata_path)

    if debug:
        logger.info("DEBUG MODE: Truncating datasets.")
        train_df = train_df.head(20)
        val_df = val_df.head(10)

    # 3. Cache Image Paths
    # We cache paths for both train and val to avoid filesystem bottlenecks
    logger.info("Caching image paths...")
    train_paths_map = cache_image_paths(
        train_df, "train", load_cached_data=load_cached_data
    )
    val_paths_map = cache_image_paths(val_df, "val", load_cached_data=load_cached_data)

    # 4. Datasets & Dataloaders
    train_dataset = RSNADataset(
        train_df,
        train_paths_map,
        phase="train",
        transform=get_transforms("train"),
    )
    val_dataset = RSNADataset(
        val_df,
        val_paths_map,
        phase="val",
        transform=get_transforms("val"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 5. Model Initialization
    logger.info(f"Initializing model: {Config.backbone}")
    model = CervicalSpineModel()
    model.to(Config.device)

    # 6. Optimizer, Scheduler, Loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.min_lr
    )

    criterion = WeightedMultiLabelLoss()
    scaler = GradScaler()  # For Mixed Precision

    # 7. Training Loop
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        logger.info(f"=== Epoch {epoch + 1}/{epochs} ===")

        # Train
        train_loss = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scaler,
            epoch,
            logger,
            Config.accum_iter,
        )

        # Validate
        val_loss = validate(val_loader, model, criterion, logger)

        # Step Scheduler
        scheduler.step()
        logger.info(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")

        # Checkpointing & Early Stopping
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            patience_counter = 0
            logger.info(f"New best model found! Loss: {best_loss}")
        else:
            patience_counter += 1
            logger.info(
                f"No improvement. Patience: {patience_counter}/{Config.patience}"
            )

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_loss": best_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
        )

        # Early Stopping
        if patience_counter >= Config.patience:
            logger.info("Early stopping triggered. Training finished.")
            break

    logger.info(f"Training complete. Best Validation Loss: {best_loss}")
