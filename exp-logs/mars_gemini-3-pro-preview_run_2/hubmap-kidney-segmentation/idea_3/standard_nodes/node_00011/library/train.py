import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import seed_everything
from library.loss import BCEDiceLoss
from library.model import AttentionUNetResNet34
from library.dataset import get_dataloader


def compute_dice_score(logits, targets, threshold=0.5):
    """
    Computes the Dice coefficient for a batch of predictions.
    Args:
        logits (torch.Tensor): Raw model outputs (before sigmoid).
        targets (torch.Tensor): Ground truth binary masks.
        threshold (float): Threshold to convert probabilities to binary mask.
    Returns:
        float: Dice score for the batch.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    # Add small epsilon to avoid division by zero
    epsilon = 1e-7
    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    return dice.item()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Runs validation on the validation set.
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            # Calculate Dice score for this batch
            batch_dice = compute_dice_score(
                outputs, masks, threshold=Config.PREDICTION_THRESHOLD
            )
            running_dice += batch_dice * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_dice = running_dice / len(dataloader.dataset)

    return epoch_loss, epoch_dice


def run_training():
    """
    Main orchestration function for training the model.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Loaders
    print("Initializing dataloaders...")
    train_loader = get_dataloader(phase="train", load_cached_data=True)
    val_loader = get_dataloader(phase="val", load_cached_data=True)

    # 3. Model
    print("Initializing Anatomy-Aware Attention U-Net (ResNet34)...")
    model = AttentionUNetResNet34(
        in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES, pretrained=True
    )
    model = model.to(device)

    # 4. Loss, Optimizer, Scheduler
    criterion = BCEDiceLoss(
        bce_weight=Config.LOSS_BCE_WEIGHT, dice_weight=Config.LOSS_DICE_WEIGHT
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 5. Training Loop Variables
    best_dice = -1.0
    patience_counter = 0
    start_time = time.time()

    print("Starting training loop...")
    print(f"Total Epochs: {Config.EPOCHS}")
    print(f"Warmup Epochs: {Config.WARMUP_EPOCHS}")
    print(f"Early Stopping Patience: {Config.EARLY_STOPPING_PATIENCE}")

    for epoch in range(1, Config.EPOCHS + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_duration = time.time() - epoch_start

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | Time: {epoch_duration:.2f}s | LR: {current_lr}"
        )
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss: {val_loss}")
        print(f"  Val Dice: {val_dice}")

        # Checkpointing and Early Stopping Logic
        is_best = False
        if val_dice > best_dice:
            best_dice = val_dice
            is_best = True
            patience_counter = 0  # Reset patience on improvement

            # Save Best Model
            print(
                f"  New best Dice score! Saving model to {Config.MODEL_CHECKPOINT_PATH}"
            )
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
        else:
            # Only increment patience if we are past the warmup period
            if epoch > Config.WARMUP_EPOCHS:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

        # Early Stopping Trigger
        if (
            epoch > Config.WARMUP_EPOCHS
            and patience_counter >= Config.EARLY_STOPPING_PATIENCE
        ):
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s. Best Val Dice: {best_dice}")
