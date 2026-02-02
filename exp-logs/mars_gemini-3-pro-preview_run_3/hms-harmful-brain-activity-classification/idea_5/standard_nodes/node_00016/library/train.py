import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

import library.config as config
from library.dataset import HMSDataset
from library.model import HybridModel
from library.utils import (
    seed_everything,
    AverageMeter,
    KLDivLossWithLogits,
    save_checkpoint,
)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()
    start_time = time.time()

    for batch_idx, (eeg, spec, targets) in enumerate(loader):
        eeg = eeg.to(device, non_blocking=True)
        spec = spec.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        logits = model(eeg, spec)

        # Calculate loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

        optimizer.step()

        losses.update(loss.item(), eeg.size(0))

    elapsed = time.time() - start_time
    print(f"Epoch {epoch} [Train] Loss: {losses.avg} | Time: {elapsed:.2f}s")
    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (eeg, spec, targets) in enumerate(loader):
            eeg = eeg.to(device, non_blocking=True)
            spec = spec.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(eeg, spec)
            loss = criterion(logits, targets)

            losses.update(loss.item(), eeg.size(0))

    elapsed = time.time() - start_time
    print(f"Epoch N/A [Val] Loss: {losses.avg} | Time: {elapsed:.2f}s")
    return losses.avg


def train(debug=False, load_cached_data=True):
    """
    Main training routine.

    Args:
        debug (bool): If True, runs on a small subset of data for quick testing.
        load_cached_data (bool): Whether to load pre-processed .npy files.
    """
    seed_everything(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # ==========================
    # Data Loading
    # ==========================
    print("Initializing Datasets...")
    train_dataset = HMSDataset(
        csv_file=config.TRAIN_META_PATH,
        mode="train",
        augment=True,
        load_cached_data=load_cached_data,
    )
    val_dataset = HMSDataset(
        csv_file=config.VAL_META_PATH,
        mode="val",
        augment=False,
        load_cached_data=load_cached_data,
    )

    if debug:
        print("DEBUG MODE: Subsetting datasets...")
        train_indices = np.arange(min(len(train_dataset), 200))  # 200 samples
        val_indices = np.arange(min(len(val_dataset), 50))  # 50 samples
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # ==========================
    # Model Setup
    # ==========================
    print("Initializing Model...")
    model = HybridModel()
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing Warm Restarts
    # T_0 is number of epochs for the first restart
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.EPOCHS, eta_min=1e-6
    )

    # Loss Function
    criterion = KLDivLossWithLogits(reduction="batchmean")

    # ==========================
    # Training Loop
    # ==========================
    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {config.EPOCHS} epochs...")

    for epoch in range(1, config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.EPOCHS} ---")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch} Summary: Train Loss={train_loss}, Val Loss={val_loss}, LR={current_lr}"
        )

        # Checkpointing & Early Stopping
        if val_loss < (best_loss - config.EARLY_STOPPING_MIN_DELTA):
            print(
                f"Validation loss improved from {best_loss} to {val_loss}. Saving model..."
            )
            best_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch, val_loss, config.MODEL_PATH
            )
        else:
            patience_counter += 1
            print(
                f"Validation loss did not improve. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}"
            )

            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation Loss: {best_loss}")
