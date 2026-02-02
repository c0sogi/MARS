import os
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import seed_everything, MetricMonitor, save_checkpoint
from library.dataset import prepare_data
from library.model import WideDeepBiLSTM


def weighted_l1_loss(pred, target, u_out, config):
    """
    Calculates L1 loss with different weights for inspiratory and expiratory phases.
    Inspiratory (u_out=0): Weight = 1.0
    Expiratory (u_out=1): Weight = 0.1
    """
    # u_out is 0 for inspiratory, 1 for expiratory
    # weights: 1.0 where u_out=0, 0.1 where u_out=1
    weights = 1.0 - (1.0 - config.W_EXPIRATORY) * u_out

    loss = torch.abs(pred - target) * weights
    return loss.mean()


def train_one_epoch(model, loader, optimizer, scheduler, device, config):
    model.train()
    metric_monitor = MetricMonitor()

    for batch in loader:
        X = batch["X"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        u_out = batch["u_out"].to(device, non_blocking=True)

        optimizer.zero_grad()

        pred = model(X)
        loss = weighted_l1_loss(pred, y, u_out, config)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        metric_monitor.update("Loss", loss.item())

    return metric_monitor.get_avg("Loss")


def validate_one_epoch(model, loader, device, config):
    model.eval()
    metric_monitor = MetricMonitor()

    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            u_out = batch["u_out"].to(device, non_blocking=True)

            pred = model(X)

            # Competition Metric: MAE on inspiratory phase only (u_out == 0)
            mask = u_out == 0

            # Avoid NaN if batch has no inspiratory phase (unlikely but safe)
            if mask.sum() > 0:
                mae = torch.abs(pred[mask] - y[mask]).mean()
                metric_monitor.update("MAE", mae.item(), n=mask.sum().item())

    return metric_monitor.get_avg("MAE")


def run_training(config=None, debug=False, limit_breaths=None):
    if config is None:
        config = Config()

    seed_everything(config.SEED)

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare Data
    # We use cached data if available, otherwise process from scratch
    train_loader = prepare_data(
        split="train",
        config=config,
        load_cached_data=True,
        debug=debug,
        limit_breaths=limit_breaths,
    )

    val_loader = prepare_data(
        split="val",
        config=config,
        load_cached_data=True,
        debug=debug,
        limit_breaths=limit_breaths,
    )

    # Determine Input Dimension from a sample batch
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch["X"].shape[-1]

    # Initialize Model
    model = WideDeepBiLSTM(input_dim=input_dim, config=config)
    model = model.to(device)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    # steps_per_epoch is required for OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # Training Loop with Early Stopping
    best_mae = float("inf")
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    patience = 7
    patience_counter = 0

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, config
        )
        val_mae = validate_one_epoch(model, val_loader, device, config)

        print(f"Epoch {epoch} | Train Loss: {train_loss} | Val MAE: {val_mae}")

        if val_mae < best_mae:
            best_mae = val_mae
            patience_counter = 0
            save_checkpoint(model.state_dict(), best_model_path)
            print(f"New best model saved with Val MAE: {best_mae}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best Val MAE: {best_mae}"
                )
                break

    print(f"Training finished. Best Validation MAE: {best_mae}")
    return best_mae
