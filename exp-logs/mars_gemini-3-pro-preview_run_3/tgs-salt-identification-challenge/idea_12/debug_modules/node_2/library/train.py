import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import seed_everything, do_kaggle_metric
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.model import UNetPlusPlus
from library.dataset import get_fold_loaders


def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    """
    Trains the model for one epoch using Automatic Mixed Precision.
    Handles Deep Supervision outputs (list of tensors).
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for inputs, targets in loader:
        batch_size = inputs.size(0)
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast():
            # UNet++ returns a list of outputs [out1, out2, out3, out4] in training mode
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def validate(model, loader, device):
    """
    Validates the model on the validation set.
    Performs center cropping to revert padding (128x128 -> 101x101) before metric calculation.
    """
    model.eval()
    preds = []
    truths = []

    # Calculate crop indices to revert padding
    # Albumentations PadIfNeeded(min_height, min_width) centers the image by default
    h_start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    w_start = (Config.IMG_SIZE - Config.ORIG_SIZE) // 2
    h_end = h_start + Config.ORIG_SIZE
    w_end = w_start + Config.ORIG_SIZE

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # UNet++ returns a single tensor (the final head) in eval mode
            output = model(inputs)
            output = torch.sigmoid(output)

            # Crop prediction and target back to original size (101x101)
            # This eliminates padding artifacts from the metric calculation
            output = output[:, :, h_start:h_end, w_start:w_end]
            targets = targets[:, :, h_start:h_end, w_start:w_end]

            preds.append(output.cpu().numpy())
            truths.append(targets.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    truths = np.concatenate(truths, axis=0)

    # Remove channel dimension: (N, 1, H, W) -> (N, H, W)
    preds = preds.squeeze(1)
    truths = truths.squeeze(1)

    # Calculate mAP over thresholds [0.5, ..., 0.95]
    score = do_kaggle_metric(preds, truths)
    return score


def train_fold(fold_idx, debug=False):
    """
    Orchestrates the training process for a single fold.

    Args:
        fold_idx (int): The fold index (0-4).
        debug (bool): If True, runs for fewer epochs (Config.DEBUG_EPOCHS).

    Returns:
        float: The best mAP score achieved.
    """
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)

    # Get DataLoaders
    # Uses caching mechanism implemented in dataset.py
    train_loader, val_loader = get_fold_loaders(fold_idx, load_cached_data=True)

    # Initialize Model
    model = UNetPlusPlus()
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Mixed Precision Scaler
    scaler = GradScaler()

    # Scheduler
    # Monitors mAP (max mode)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    # Loss Functions for Curriculum
    criterion_bce = BCEDiceLoss()
    criterion_lovasz = LovaszHingeLoss()

    # Training State
    best_score = 0.0
    patience_counter = 0
    epochs = Config.DEBUG_EPOCHS if debug else Config.EPOCHS

    print(f"Starting training for Fold {fold_idx} on {device}...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Loss Curriculum: Warmup with BCE+Dice, Finetune with Lovasz
        if epoch <= Config.WARMUP_EPOCHS:
            criterion = criterion_bce
            loss_name = "BCE+Dice"
        else:
            criterion = criterion_lovasz
            loss_name = "Lovasz"

        # Train Step
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device
        )

        # Validation Step
        val_score = validate(model, val_loader, device)

        elapsed = time.time() - start_time

        # Print metrics (Full precision for val_score as requested)
        print(
            f"Epoch {epoch}/{epochs} [{loss_name}] - "
            f"Train Loss: {train_loss:.6f}, "
            f"Val mAP: {val_score}, "
            f"Time: {elapsed:.0f}s"
        )

        # Update Scheduler
        scheduler.step(val_score)

        # Save Checkpoint
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0

            save_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold_idx}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best score! Saved model to {save_path}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Fold {fold_idx} finished. Best mAP: {best_score}")

    # Clear memory
    del model, optimizer, scaler, train_loader, val_loader
    torch.cuda.empty_cache()

    return best_score
