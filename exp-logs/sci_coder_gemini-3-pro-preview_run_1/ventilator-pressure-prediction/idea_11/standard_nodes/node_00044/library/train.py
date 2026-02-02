import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.model import VentilatorModel
from library.dataset import get_data_loaders
from library.utils import seed_everything, masked_mae_metric


def compute_masked_loss(
    pred: torch.Tensor, target: torch.Tensor, u_out: torch.Tensor
) -> torch.Tensor:
    """
    Computes the L1 loss masked by the inspiratory phase (u_out == 0).
    """
    mask = 1 - u_out
    loss = torch.abs(pred - target) * mask
    # Sum of errors / Sum of mask (count of valid steps)
    # Add epsilon to denominator to prevent division by zero
    return loss.sum() / (mask.sum() + 1e-8)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.OneCycleLR,
    device: torch.device,
    aux_weight: float,
) -> float:
    """
    Runs one epoch of training.
    Returns the average training loss.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (X, y, u_out) in enumerate(loader):
        X, y, u_out = X.to(device), y.to(device), u_out.to(device)

        optimizer.zero_grad()

        # Forward pass
        final_pred, aux_pred = model(X)

        # Compute losses
        loss_final = compute_masked_loss(final_pred, y, u_out)

        loss = loss_final

        # Add auxiliary loss if available
        if aux_pred is not None:
            loss_aux = compute_masked_loss(aux_pred, y, u_out)
            loss = loss + aux_weight * loss_aux

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended for LSTMs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """
    Runs validation.
    Returns the average Masked MAE metric.
    """
    model.eval()
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for X, y, u_out in loader:
            X, y, u_out = X.to(device), y.to(device), u_out.to(device)

            # Forward pass (ignore aux head for validation metric)
            final_pred, _ = model(X)

            # Calculate metric
            mae = masked_mae_metric(final_pred, y, u_out)

            total_mae += mae
            num_batches += 1

    return total_mae / num_batches


def run_training(
    debug_limit: int = None,
    load_cached_data: bool = True,
    save_path: str = "model.pth",
):
    """
    Main training function.

    Args:
        debug_limit (int, optional): Limit dataset size for debugging.
        load_cached_data (bool): Whether to use cached preprocessed data.
        save_path (str): Filename to save the best model.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    full_save_path = os.path.join(Config.WORKING_DIR, save_path)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_data_loaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
        debug_limit=debug_limit,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = VentilatorModel(config=Config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR_MAX, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR requires total steps
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * Config.EPOCHS

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR_MAX,
        total_steps=total_steps,
        pct_start=Config.PCT_START,
        anneal_strategy="cos",
    )

    # 5. Training Loop
    best_val_mae = float("inf")
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            aux_weight=Config.AUX_LOSS_WEIGHT,
        )

        # Validate
        val_mae = validate(model, val_loader, device)

        elapsed = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val MAE: {val_mae}"
        )

        # Checkpointing
        if val_mae < best_val_mae:
            print(f"New best Val MAE! ({best_val_mae} -> {val_mae}). Saving model...")
            best_val_mae = val_mae
            torch.save(model.state_dict(), full_save_path)

    print(f"Training complete. Best Validation MAE: {best_val_mae}")
    print(f"Best model saved to: {full_save_path}")
