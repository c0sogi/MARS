import os
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import seed_everything, AverageMeter, kl_divergence_loss
from library.data import get_dataloaders
from library.model import DualStreamNetwork


def train_one_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    """
    Trains the model for one epoch using the provided loader and optimizer.
    Handles mixed precision training if a scaler is provided.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Unpack inputs: (eeg_tensor, spec_tensor)
        eeg, spec = inputs
        eeg = eeg.to(device, non_blocking=True)
        spec = spec.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed Precision Context
        if scaler:
            with torch.amp.autocast(device.type):
                logits = model((eeg, spec))
                loss = kl_divergence_loss(logits, targets)

            # Scale loss and backward
            scaler.scale(loss).backward()

            # Unscale for gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Step optimizer and updater scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training
            logits = model((eeg, spec))
            loss = kl_divergence_loss(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
            optimizer.step()

        # Step scheduler (OneCycleLR updates every batch)
        if scheduler is not None:
            scheduler.step()

        loss_meter.update(loss.item(), eeg.size(0))

    return loss_meter.avg


def validate_one_epoch(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for inputs, targets in loader:
            eeg, spec = inputs
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            logits = model((eeg, spec))
            loss = kl_divergence_loss(logits, targets)

            loss_meter.update(loss.item(), eeg.size(0))

    return loss_meter.avg


def run_training(debug=Config.DEBUG, epochs=Config.EPOCHS):
    """
    Main training loop.
    Initializes model, data, and optimizer, then runs the training/val loop
    with early stopping and checkpointing.
    """
    # 1. Setup Environment
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    # 2. Prepare Data
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # 3. Initialize Model
    model = DualStreamNetwork().to(device)

    # 4. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR configuration
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 5. Mixed Precision Scaler
    scaler = None
    if Config.MIXED_PRECISION and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    # 6. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {epochs} epochs...")
    if debug:
        print("Debug mode enabled.")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler
        )

        # Validate
        val_loss = validate_one_epoch(model, val_loader, device)

        # Logging (Full precision)
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Loss: {best_val_loss}")

    # Reload best model weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    return model
