import os
import time
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import set_seed, MetricMonitor, dice_coef
from library.loss import BCETverskyLoss
from library.model import UNetEfficientNet
from library.data import UWDataset, get_transforms, prepare_data


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Handles the training of a single epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    # Iterate over batches
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # Forward pass
        y_pred = model(images)

        # Compute loss
        loss = criterion(y_pred, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        metric_monitor.update("Loss", loss.item())

    return metric_monitor.get("Loss")


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes Loss and Mean Dice Coefficient.
    """
    model.eval()
    metric_monitor = MetricMonitor()

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            y_pred = model(images)

            # Compute loss
            loss = criterion(y_pred, masks)

            # Compute Dice
            # Apply sigmoid to get probabilities
            y_pred_prob = torch.sigmoid(y_pred)
            # Thresholding for binary mask calculation
            y_pred_mask = (y_pred_prob > Config.PRED_THR).float()

            dice = dice_coef(masks, y_pred_mask)

            # Update metrics
            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Dice", dice)

    return metric_monitor.get("Loss"), metric_monitor.get("Dice")


def run_training(debug=Config.DEBUG, load_cached_data=True):
    """
    Main driver function to setup data, model, and run the training loop.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Create output directories
    Config.setup()

    # 2. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    # 3. Prepare Data (2.5D Context & Caching)
    # This adds prev_path/next_path columns and caches the result to parquet
    df_train = prepare_data(df_train, load_cached_data=load_cached_data, split="train")
    df_val = prepare_data(df_val, load_cached_data=load_cached_data, split="val")

    # Debugging: Subsample data
    if debug:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        df_train = df_train.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)
        df_val = df_val.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

    print(f"Training on {len(df_train)} samples, Validating on {len(df_val)} samples.")

    # 4. Datasets & Dataloaders
    train_dataset = UWDataset(
        df_train, transforms=get_transforms(data="train"), mode="train"
    )
    val_dataset = UWDataset(
        df_val, transforms=get_transforms(data="valid"), mode="valid"
    )

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
    )

    # 5. Model Initialization
    model = UNetEfficientNet(
        backbone_name=Config.BACKBONE, pretrained=True, classes=Config.NUM_CLASSES
    )
    model = model.to(device)

    # 6. Loss, Optimizer, Scheduler
    criterion = BCETverskyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
    )
    scaler = GradScaler()

    # 7. Training Loop
    best_dice = -np.inf
    best_epoch = -1
    patience = 5  # Early stopping patience
    patience_counter = 0

    print("\nStarting Training...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        # Inline train_one_epoch logic to support AMP
        model.train()
        metric_monitor = MetricMonitor()

        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            # Forward pass with AMP
            with autocast():
                y_pred = model(images)
                loss = criterion(y_pred, masks)

            # Backward pass with Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update metrics
            metric_monitor.update("Loss", loss.item())

        train_loss = metric_monitor.get("Loss")

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss:.10f} | "
            f"Val Loss: {val_loss:.10f} | Val Dice: {val_dice:.10f}"
        )

        # Model Checkpointing
        if val_dice > best_dice:
            print(
                f"Validation Dice Improved ({best_dice:.6f} ---> {val_dice:.6f}). Saving model..."
            )
            best_dice = val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(
        f"\nTraining Complete. Best Validation Dice: {best_dice:.10f} at Epoch {best_epoch}"
    )
    print(f"Best model saved to: {Config.CHECKPOINT_PATH}")
