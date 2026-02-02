import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from library.config import Config
from library.utils import set_seed, get_logger
from library.dataset import ContrailDataset, get_transforms
from library.model import DualStreamUNet
from library.loss import HybridLoss


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks, _) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        with autocast():
            logits = model(images)
            loss = loss_fn(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, loss_fn, device):
    """
    Runs validation and computes Global Dice Score.
    Global Dice = 2 * (Total Intersection) / (Total Union) over the dataset.
    """
    model.eval()
    running_loss = 0.0

    # Accumulators for Global Dice
    intersection_sum = 0.0
    union_sum = 0.0

    # Threshold for binarization during metric calculation
    threshold = Config.THRESHOLD

    with torch.no_grad():
        for images, masks, _ in enumerate(loader):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            with autocast():
                logits = model(images)
                loss = loss_fn(logits, masks)

            running_loss += loss.item()

            # Metric Calculation
            # Apply sigmoid and threshold
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # Flatten for calculation
            preds = preds.view(-1)
            targets = masks.view(-1)

            intersection_sum += (preds * targets).sum().item()
            union_sum += preds.sum().item() + targets.sum().item()

    avg_loss = running_loss / len(loader)

    # Compute Global Dice
    # Handle edge case where union is 0 (both empty) -> score 1.0
    if union_sum == 0:
        global_dice = 1.0
    else:
        global_dice = (2.0 * intersection_sum) / union_sum

    return avg_loss, global_dice


def train_model(debug=False):
    """
    Main training function.
    """
    # 1. Setup
    set_seed(Config.SEED)
    logger = get_logger("train")
    device = torch.device(Config.DEVICE)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    logger.info(f"Starting training with device: {device}")
    logger.info(f"Debug mode: {debug}")

    # 2. Data Loading
    train_dataset = ContrailDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        split="train",
        transform=get_transforms("train"),
        debug=debug,
    )

    valid_dataset = ContrailDataset(
        metadata_path=Config.VALID_METADATA_PATH,
        split="validation",
        transform=get_transforms("validation"),
        debug=debug,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    logger.info(
        f"Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}"
    )

    # 3. Model & Optimization
    model = DualStreamUNet(
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        in_chans_a=Config.IN_CHANNELS_STREAM_A,
        in_chans_b=Config.IN_CHANNELS_STREAM_B,
    )
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Hybrid Loss: BCE + BatchDice
    loss_fn = HybridLoss(bce_weight=1.0, dice_weight=1.0)

    scaler = GradScaler()

    # 4. Training Loop
    best_dice = 0.0
    patience = 10  # Early stopping patience
    patience_counter = 0

    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler
        )

        # Validate
        val_loss, val_dice = validate(model, valid_loader, loss_fn, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start

        # Logging
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {epoch_duration:.2f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val Global Dice: {val_dice:.16f}"
        )

        # Checkpointing & Early Stopping
        if val_dice > best_dice:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            logger.info(f"New best model saved with Global Dice: {best_dice:.16f}")
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    logger.info(
        f"Training completed in {total_time:.2f}s. Best Global Dice: {best_dice:.16f}"
    )
