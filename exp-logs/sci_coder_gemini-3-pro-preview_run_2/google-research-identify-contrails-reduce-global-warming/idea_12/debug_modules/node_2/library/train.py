import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_train_val_loaders
from library.model import ConvNeXtUNet
from library.loss import HybridLoss


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using the Global Dice metric.
    Global Dice = 2 * (Total Intersection) / (Total Union) over all data.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # Accumulators for Global Dice
    total_intersection = 0.0
    total_union = 0.0

    # Threshold for binarization
    threshold = Config.THRESHOLD

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            batch_size = images.size(0)

            with autocast(enabled=True):
                logits = model(images)
                loss = criterion(logits, masks)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Calculate Global Dice components
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # Flatten to compute intersection/union over the batch volume
            preds_flat = preds.view(-1)
            targets_flat = masks.view(-1)

            intersection = (preds_flat * targets_flat).sum().item()
            union = preds_flat.sum().item() + targets_flat.sum().item()

            total_intersection += intersection
            total_union += union

    val_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

    # Compute Global Dice
    # Add epsilon to avoid division by zero if both sets are empty
    smooth = 1e-6
    global_dice = (2.0 * total_intersection + smooth) / (total_union + smooth)

    return val_loss, global_dice


def train_loop(epochs=Config.EPOCHS, patience=10):
    """
    Main training loop with Early Stopping and Model Checkpointing.
    """
    seed_everything(Config.SEED)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # Load Data
    print("Loading datasets...")
    train_loader, val_loader = get_train_val_loaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Initialize Model
    print("Initializing model...")
    model = ConvNeXtUNet()
    model.to(device)

    # Optimizer, Scheduler, Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = HybridLoss()
    scaler = GradScaler()

    # Training State
    best_dice = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start

        # Logging (Full precision for metrics)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Time: {epoch_duration:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Global Dice: {val_dice:.10f}"
        )

        # Checkpointing
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice:.6f} to {val_dice:.6f}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in Dice. Patience: {epochs_no_improve}/{patience}")

        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    total_time = time.time() - start_time
    print(
        f"Training finished in {total_time:.1f}s. Best Validation Global Dice: {best_dice:.10f}"
    )
