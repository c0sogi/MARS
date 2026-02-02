import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.dataset import load_metadata_df, get_transforms, ArtworkDataset
from library.model import ArtworkModel
from library.utils import ModelEMA, calculate_f1, save_checkpoint, seed_everything


def train_one_epoch(
    model, loader, optimizer, criterion, device, scaler, ema_model=None
):
    """
    Executes one epoch of training.
    """
    model.train()
    total_loss = 0.0
    num_steps = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Apply Label Smoothing manually to targets
        # Target transformation: new_target = target * (1 - eps) + 0.5 * eps
        if Config.LABEL_SMOOTHING > 0:
            targets = (
                targets * (1.0 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
            )

        optimizer.zero_grad()

        with autocast(enabled=Config.USE_AMP):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()

        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        if ema_model:
            ema_model.update(model)

        total_loss += loss.item()
        num_steps += 1

    return total_loss / num_steps if num_steps > 0 else 0.0


@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    total_loss = 0.0
    num_steps = 0

    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=Config.USE_AMP):
            logits = model(images)
            loss = criterion(logits, targets)

        total_loss += loss.item()
        num_steps += 1

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        all_preds.append(probs.cpu())
        all_targets.append(targets.cpu())

    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0

    # Concatenate all batches
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    return avg_loss, all_preds, all_targets


def find_best_threshold(preds, targets):
    """
    Performs a grid search to find the threshold that maximizes Micro F1 score.
    """
    best_score = 0.0
    best_thresh = 0.5

    # Generate thresholds from config range
    thresholds = np.arange(
        Config.THRESHOLD_SEARCH_START,
        Config.THRESHOLD_SEARCH_END + 1e-6,
        Config.THRESHOLD_SEARCH_STEP,
    )

    # Iterate through thresholds
    for thresh in thresholds:
        binary_preds = (preds > thresh).astype(int)
        score = calculate_f1(binary_preds, targets)

        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_score, best_thresh


def run_training(load_cached_data=True, save_checkpoint_name="best_model.pth"):
    """
    Main training pipeline.
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # 2. Prepare Data
    print("Loading and processing metadata...")
    train_df = load_metadata_df("train", load_cached_data=load_cached_data)
    val_df = load_metadata_df("val", load_cached_data=load_cached_data)

    # Initialize Datasets
    train_dataset = ArtworkDataset(
        train_df, mode="train", transforms=get_transforms("train", Config.IMG_SIZE)
    )
    val_dataset = ArtworkDataset(
        val_df, mode="val", transforms=get_transforms("val", Config.IMG_SIZE)
    )

    # Initialize Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
    )

    # 3. Initialize Model
    print(f"Initializing model: {Config.MODEL_NAME}")
    model = ArtworkModel(pretrained=True)
    model.to(device)

    # Initialize EMA
    ema_model = None
    if Config.USE_EMA:
        print("Initializing ModelEMA...")
        ema_model = ModelEMA(model, decay=Config.EMA_DECAY)

    # 4. Setup Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR)

    scaler = GradScaler(enabled=Config.USE_AMP)

    # Loss Function with Positive Weighting
    # We create a weight tensor for BCEWithLogitsLoss
    pos_weight = torch.ones([Config.NUM_CLASSES], device=device) * Config.POS_WEIGHT
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 5. Training Loop
    best_f1 = 0.0

    print(f"Training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train Step
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, ema_model
        )

        # Scheduler Step
        scheduler.step()

        # Validation Step
        # Use EMA model for validation if available, otherwise standard model
        val_model_to_use = ema_model.ema if ema_model else model
        val_loss, val_preds, val_targets = validate(
            val_model_to_use, val_loader, criterion, device
        )

        # Threshold Tuning
        current_f1, best_thresh = find_best_threshold(val_preds, val_targets)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"F1: {current_f1:.8f} (Thresh: {best_thresh:.2f})"
        )

        # Save Best Model
        if current_f1 > best_f1:
            best_f1 = current_f1
            print(f"New Best F1 Score! Saving checkpoint to {save_checkpoint_name}")
            save_checkpoint(
                val_model_to_use,  # Save the weights that produced the score
                optimizer,
                scheduler,
                epoch,
                best_f1,
                filename=save_checkpoint_name,
            )

    print(f"Training finished. Best Validation F1: {best_f1:.8f}")
