import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    save_checkpoint,
    compute_metric,
)
from library.dataset import get_data_loaders
from library.model import RDHNet


def train_epoch(model, loader, optimizer, device, config):
    """
    Performs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()

    for batch_idx, batch_data in enumerate(loader):
        # Move data to device
        x = batch_data["x"].to(device)
        u_out = batch_data["u_out"].to(device)
        y = batch_data["y"].to(device)

        # Forward pass
        # Output shape: (Batch, Seq_Len, 1)
        preds = model(x)

        # Flatten for loss calculation
        preds_flat = preds.view(-1)
        y_flat = y.view(-1)
        u_out_flat = u_out.view(-1)

        # Create mask for inspiratory phase (u_out == 0)
        mask = u_out_flat == 0

        # Calculate Masked L1 Loss
        # We only compute loss on the inspiratory phase
        if mask.sum() > 0:
            loss = nn.L1Loss()(preds_flat[mask], y_flat[mask])
        else:
            # Fallback for safety, though unlikely with full breath sequences
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping (Crucial for LSTM stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)

        optimizer.step()

        # Update metrics
        batch_size = x.size(0)
        loss_meter.update(loss.item(), batch_size)

        # Calculate MAE for monitoring (reuse compute_metric logic inline or call it)
        with torch.no_grad():
            mae = compute_metric(preds_flat, y_flat, u_out_flat)
            mae_meter.update(mae, batch_size)

    return loss_meter.avg, mae_meter.avg


def validate_epoch(model, loader, device, config):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            x = batch_data["x"].to(device)
            u_out = batch_data["u_out"].to(device)
            y = batch_data["y"].to(device)

            preds = model(x)

            preds_flat = preds.view(-1)
            y_flat = y.view(-1)
            u_out_flat = u_out.view(-1)

            mask = u_out_flat == 0

            if mask.sum() > 0:
                loss = nn.L1Loss()(preds_flat[mask], y_flat[mask])
            else:
                loss = torch.tensor(0.0, device=device)

            loss_meter.update(loss.item(), x.size(0))

            # Compute competition metric
            mae = compute_metric(preds_flat, y_flat, u_out_flat)
            mae_meter.update(mae, x.size(0))

    return loss_meter.avg, mae_meter.avg


def run_training(config=None):
    """
    Main training pipeline.
    """
    if config is None:
        config = Config()

    # 1. Setup
    seed_everything(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Starting training on device: {device}")
    config.display()

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # 3. Model Initialization
    model = RDHNet(config).to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
    )

    # 5. Training Loop
    best_mae = float("inf")
    early_stopping_counter = 0

    print("\nBeginning training loop...")

    for epoch in range(1, config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, device, config
        )

        # Validate
        val_loss, val_mae = validate_epoch(model, val_loader, device, config)

        # Scheduler Step
        scheduler.step(val_mae)

        elapsed = time.time() - start_time

        # Logging
        # Printing full precision as requested
        print(
            f"Epoch {epoch}/{config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Train MAE: {train_mae} | "
            f"Val Loss: {val_loss} | "
            f"Val MAE: {val_mae}"
        )

        # Checkpointing
        is_best = val_mae < best_mae
        if is_best:
            best_mae = val_mae
            early_stopping_counter = 0
            print(f"New best model found! MAE: {best_mae}")
        else:
            early_stopping_counter += 1

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mae": best_mae,
            },
            is_best,
            config.WORKING_DIR,
        )

        # Early Stopping
        if early_stopping_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"\nTraining complete. Best Validation MAE: {best_mae}")
    return model
