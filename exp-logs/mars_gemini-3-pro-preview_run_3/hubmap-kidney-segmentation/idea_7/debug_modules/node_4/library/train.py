import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import set_seed
from library.dataset import HubmapDataset
from library.model import build_model

# -------------------------------------------------------------------------
# Helper Classes & Functions
# -------------------------------------------------------------------------


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dice_coef(y_pred, y_true, thr=0.5, epsilon=1e-7):
    """
    Computes Dice Coefficient.
    y_pred: Logits (B, 1, H, W)
    y_true: Binary Mask (B, 1, H, W)
    """
    y_pred = (torch.sigmoid(y_pred) > thr).float()
    y_true = y_true.float()

    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)

    return (2.0 * intersection) / (union + epsilon)


class CombinedLoss(nn.Module):
    """
    Combined BCEWithLogitsLoss and Dice Loss.
    """

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # BCE Loss
        bce_loss = self.bce(y_pred, y_true)

        # Dice Loss
        pred_sigmoid = torch.sigmoid(y_pred)
        intersection = (pred_sigmoid * y_true).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, deep_supervision_weights
):
    model.train()
    losses = AverageMeter()

    for images, masks in loader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)

            # Deep Supervision: outputs is a list
            if isinstance(outputs, list):
                loss = 0
                # Calculate weighted loss for each scale
                # Note: In SMP Unet++, all outputs are upsampled to input size
                for output, weight in zip(outputs, deep_supervision_weights):
                    loss += weight * criterion(output, masks)
            else:
                loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    dices = AverageMeter()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            with autocast():
                outputs = model(images)

                # For validation, we focus on the final output (index 0)
                if isinstance(outputs, list):
                    final_output = outputs[0]
                else:
                    final_output = outputs

                loss = criterion(final_output, masks)
                dice = dice_coef(final_output, masks)

            losses.update(loss.item(), images.size(0))
            dices.update(dice.item(), images.size(0))

    return losses.avg, dices.avg


# -------------------------------------------------------------------------
# Main Training Routine
# -------------------------------------------------------------------------


def train_model():
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Check if metadata exists
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    # Load Metadata
    df_train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val_full = pd.read_csv(Config.VAL_METADATA_PATH)

    # Debug Mode
    if Config.DEBUG:
        df_train_full = df_train_full.head(Config.DEBUG_SAMPLE_SIZE)
        df_val_full = df_val_full.head(Config.DEBUG_SAMPLE_SIZE)
        print(f"DEBUG MODE: Training on {len(df_train_full)} samples.")

    # Build Model
    model = build_model()
    model = model.to(device)

    # Loss Function
    criterion = CombinedLoss(bce_weight=0.5)

    # Deep Supervision Weights
    ds_weights = Config.LOSS_WEIGHTS

    # Gradient Scaler
    scaler = GradScaler()

    # Tracking Best Model
    best_dice = 0.0
    patience_counter = 0

    # ---------------------------------------------------------------------
    # Phase 1: High Throughput (Coarse)
    # ---------------------------------------------------------------------
    print("\n=== Starting Phase 1: High Throughput (512x512) ===")

    phase1_cfg = Config.PHASE1

    # Dataset & Loader for Phase 1
    train_dataset_p1 = HubmapDataset(
        df_train_full,
        phase="train",
        image_size=phase1_cfg["TILE_SIZE"],
        load_cached_data=True,
    )
    val_dataset_p1 = HubmapDataset(
        df_val_full,
        phase="validation",
        image_size=phase1_cfg["TILE_SIZE"],
        load_cached_data=True,
    )

    train_loader_p1 = DataLoader(
        train_dataset_p1,
        batch_size=phase1_cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader_p1 = DataLoader(
        val_dataset_p1,
        batch_size=phase1_cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=phase1_cfg["LR"], weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=Config.SCHEDULER_T0,
        T_mult=Config.SCHEDULER_T_MULT,
        eta_min=Config.MIN_LR,
    )

    for epoch in range(1, phase1_cfg["EPOCHS"] + 1):
        train_loss = train_one_epoch(
            model, train_loader_p1, criterion, optimizer, scaler, device, ds_weights
        )
        val_loss, val_dice = validate_one_epoch(model, val_loader_p1, criterion, device)

        scheduler.step()

        print(
            f"Phase 1 | Epoch {epoch}/{phase1_cfg['EPOCHS']} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Dice: {val_dice:.8f}"
        )

        # Checkpointing
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  >>> New Best Model Saved! Dice: {best_dice:.8f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered in Phase 1.")
            break

    # ---------------------------------------------------------------------
    # Phase 2: High Precision (Fine-Tuning)
    # ---------------------------------------------------------------------
    print("\n=== Starting Phase 2: High Precision (768x768) ===")

    # Load best model from Phase 1
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print("Loading best model from Phase 1...")
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    phase2_cfg = Config.PHASE2

    # Dataset & Loader for Phase 2
    train_dataset_p2 = HubmapDataset(
        df_train_full,
        phase="train",
        image_size=phase2_cfg["TILE_SIZE"],
        load_cached_data=True,
    )
    val_dataset_p2 = HubmapDataset(
        df_val_full,
        phase="validation",
        image_size=phase2_cfg["TILE_SIZE"],
        load_cached_data=True,
    )

    train_loader_p2 = DataLoader(
        train_dataset_p2,
        batch_size=phase2_cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader_p2 = DataLoader(
        val_dataset_p2,
        batch_size=phase2_cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Re-initialize optimizer with lower LR for fine-tuning
    optimizer = optim.AdamW(
        model.parameters(), lr=phase2_cfg["LR"], weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=Config.SCHEDULER_T0,
        T_mult=Config.SCHEDULER_T_MULT,
        eta_min=Config.MIN_LR,
    )

    # Reset patience for Phase 2
    patience_counter = 0

    for epoch in range(1, phase2_cfg["EPOCHS"] + 1):
        train_loss = train_one_epoch(
            model, train_loader_p2, criterion, optimizer, scaler, device, ds_weights
        )
        val_loss, val_dice = validate_one_epoch(model, val_loader_p2, criterion, device)

        scheduler.step()

        print(
            f"Phase 2 | Epoch {epoch}/{phase2_cfg['EPOCHS']} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Dice: {val_dice:.8f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  >>> New Best Model Saved! Dice: {best_dice:.8f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered in Phase 2.")
            break

    print(f"\nTraining Complete. Best Validation Dice: {best_dice:.8f}")
    print(f"Best model saved to: {Config.MODEL_SAVE_PATH}")
