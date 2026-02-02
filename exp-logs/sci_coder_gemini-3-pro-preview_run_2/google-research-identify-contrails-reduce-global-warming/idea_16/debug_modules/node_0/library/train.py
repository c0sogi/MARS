import os
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import set_seed, AverageMeter
from library.dataset import ContrailDataset
from library.model import ConvNeXtUNet
from library.loss import HybridLoss


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device):
    """
    Runs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, masks, _) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        with autocast(enabled=True):
            logits = model(images)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, loss_fn, device):
    """
    Runs validation and calculates the Global Dice Coefficient.
    Global Dice = 2 * (Total Intersection) / (Total Union) across all samples.
    """
    model.eval()
    losses = AverageMeter()

    # Accumulators for Global Dice
    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            with autocast(enabled=True):
                logits = model(images)
                loss = loss_fn(logits, masks)

            losses.update(loss.item(), images.size(0))

            # Compute probabilities and threshold
            preds = torch.sigmoid(logits)
            preds = (preds > Config.THRESHOLD).float()

            # Flatten for global calculation
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            intersection = (preds_flat * masks_flat).sum().item()
            union = (preds_flat + masks_flat).sum().item()

            total_intersection += intersection
            total_union += union

    # Calculate Global Dice
    # Add smooth to avoid division by zero if empty
    smooth = Config.SMOOTH
    global_dice = (2.0 * total_intersection + smooth) / (total_union + smooth)

    return losses.avg, global_dice


def train_model():
    """
    Main function to execute the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Load Metadata
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)

    # Debugging: Limit samples if configured
    if Config.MAX_TRAIN_SAMPLES:
        train_df = train_df.iloc[: Config.MAX_TRAIN_SAMPLES]
        print(f"Debugging: Limited training samples to {len(train_df)}")

    if Config.MAX_VAL_SAMPLES:
        val_df = val_df.iloc[: Config.MAX_VAL_SAMPLES]
        print(f"Debugging: Limited validation samples to {len(val_df)}")

    # 3. Datasets & Loaders
    print("Initializing datasets...")
    train_dataset = ContrailDataset(train_df, stage="train", load_cached_data=True)
    val_dataset = ContrailDataset(val_df, stage="validation", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # 4. Model, Loss, Optimizer
    print("Initializing model...")
    model = ConvNeXtUNet()
    model.to(device)

    loss_fn = HybridLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    scaler = GradScaler()

    # 5. Training Loop
    best_dice = 0.0
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, loss_fn, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Global Dice: {val_dice}"
        )

        # Save Best Model
        if val_dice > best_dice:
            print(
                f"Global Dice improved from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training complete. Best Global Dice: {best_dice}")

    # Cleanup
    del model, optimizer, scaler, scheduler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()
