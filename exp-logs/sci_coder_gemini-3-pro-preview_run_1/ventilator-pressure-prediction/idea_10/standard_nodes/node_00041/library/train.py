import os
import torch
import torch.optim as optim
import numpy as np
from torch.cuda.amp import GradScaler, autocast  # Optional mixed precision for speed

from library.config import Config
from library.utils import seed_everything
from library.model import DeepSupervisedVentilatorModel
from library.dataset import get_dataloaders
from library.loss import CompositeLoss, MaskedL1Loss


def train_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):
    """
    Performs one training epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    # Index of u_out in Config.FEATURE_COLS is 2
    # ["time_step", "u_in", "u_out", ...]
    u_out_idx = 2

    for batch_idx, (data, target) in enumerate(loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Extract u_out for masking (Batch, Seq)
        u_out = data[:, :, u_out_idx]

        optimizer.zero_grad()

        # Forward pass
        # Model returns (final_pred, aux_pred) in training mode
        final_pred, aux_pred = model(data)

        # Compute loss
        loss = criterion((final_pred, aux_pred), target, u_out)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate_epoch(model, loader, criterion, device):
    """
    Performs one validation epoch.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    u_out_idx = 2

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            u_out = data[:, :, u_out_idx]

            # Forward pass
            # Model returns only final_pred in eval mode
            pred = model(data)

            # Compute metric (Masked MAE)
            loss = criterion(pred, target, u_out)

            running_loss += loss.item()
            num_batches += 1

    return running_loss / num_batches


def run_training():
    """
    Main driver function for training the model.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Preparing dataloaders...")
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model
    print("Initializing model...")
    model = DeepSupervisedVentilatorModel().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 5. Loss Functions
    train_criterion = CompositeLoss().to(device)  # Weighted Masked L1 (Final + Aux)
    val_criterion = MaskedL1Loss().to(device)  # Pure Masked L1 (MAE)

    # Optional: Gradient Scaler for mixed precision if supported, though not strictly required
    # We'll stick to standard float32 for safety unless memory is tight, but provided code uses float32 tensors.
    scaler = None

    # 6. Training Loop
    best_val_mae = float("inf")

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, train_criterion, device, scaler
        )

        # Validate
        val_mae = validate_epoch(model, val_loader, val_criterion, device)

        # Logging
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss} | Val MAE: {val_mae}"
        )

        # Checkpointing
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            print(f"New best model found! Saving to {Config.MODEL_PATH}")
            torch.save(model.state_dict(), Config.MODEL_PATH)

    print(f"Training complete. Best Validation MAE: {best_val_mae}")


if __name__ == "__main__":
    run_training()
