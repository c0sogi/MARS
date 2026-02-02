import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, mixup_data, mixup_criterion
from library.dataset import get_dataloader
from library.model import EEGNet


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one epoch using MixUp augmentation.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = images.size(0)

        # Apply MixUp
        # We use a Beta distribution parameter alpha=1.0 (uniform) or similar
        mixed_images, targets_a, targets_b, lam = mixup_data(
            images, targets, alpha=1.0, device=device
        )

        optimizer.zero_grad()

        # Forward pass
        outputs = model(mixed_images)

        # KLDivLoss expects log-probabilities, but model outputs probabilities (Softmax)
        # Add epsilon for numerical stability
        log_outputs = torch.log(outputs + 1e-6)

        # Calculate Loss
        loss = mixup_criterion(criterion, log_outputs, targets_a, targets_b, lam)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    duration = time.time() - start_time

    print(f"Epoch {epoch} | Train Loss: {epoch_loss:.6f} | Time: {duration:.1f}s")

    return epoch_loss


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    start_time = time.time()

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch_size = images.size(0)

            # Forward pass
            outputs = model(images)

            # KLDivLoss expects log-probabilities
            log_outputs = torch.log(outputs + 1e-6)

            # Calculate Loss
            loss = criterion(log_outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    val_loss = running_loss / dataset_size
    duration = time.time() - start_time

    print(f"Validation | Loss: {val_loss} | Time: {duration:.1f}s")

    return val_loss


def train(debug_subset_size=None):
    """
    Main training loop.
    Initializes model, data, optimizer, and scheduler.
    Implements Early Stopping and saves the best model.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    device = torch.device(Config.DEVICE)

    print(f"Training on device: {device}")

    # 2. Data Loaders
    # Note: caching is handled internally by get_dataloader -> load_data
    train_loader = get_dataloader(
        mode="train",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug_subset=debug_subset_size,
    )

    val_loader = get_dataloader(
        mode="val",
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug_subset=debug_subset_size,
    )

    # 3. Model
    model = EEGNet(pretrained=True)
    model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR)

    # 5. Loss Function
    # reduction='batchmean' is mathematically correct for KL Div
    criterion = nn.KLDivLoss(reduction="batchmean")

    # 6. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{Config.EPOCHS} ---")

        # Train Step
        train_loss = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device
        )

        # Validation Step
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.2e}")

        # Checkpoint & Early Stopping
        if val_loss < best_val_loss:
            print(
                f"Validation Loss Improved ({best_val_loss} -> {val_loss}). Saving model..."
            )
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"Validation Loss did not improve. Patience: {patience_counter}/{Config.PATIENCE}"
            )

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss}")
    print(f"Best model saved to: {Config.MODEL_PATH}")
