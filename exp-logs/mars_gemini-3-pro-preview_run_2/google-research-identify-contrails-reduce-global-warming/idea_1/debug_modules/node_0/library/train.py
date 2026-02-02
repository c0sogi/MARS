import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.dataset import ContrailDataset
from library.model import MobileNetUNet
from library.utils import seed_everything, dice_coef


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Soft Dice Loss.
    """

    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred_logits, y_true):
        # BCE Loss
        bce = self.bce_loss(y_pred_logits, y_true)

        # Soft Dice Loss
        y_pred_probs = torch.sigmoid(y_pred_logits)

        # Flatten
        y_pred_flat = y_pred_probs.view(-1)
        y_true_flat = y_true.view(-1)

        intersection = (y_pred_flat * y_true_flat).sum()
        union = y_pred_flat.sum() + y_true_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        return self.bce_weight * bce + self.dice_weight * dice_loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device, threshold=0.5):
    """
    Validates the model and calculates Global Dice Coefficient.
    """
    model.eval()
    running_loss = 0.0

    # Variables for Global Dice calculation
    # intersection_sum = |X n Y|
    # union_sum = |X| + |Y|
    intersection_sum = 0.0
    union_sum = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Calculate metrics
            preds_prob = torch.sigmoid(outputs)
            preds_bin = (preds_prob > threshold).float()

            # Flatten for global accumulation
            preds_flat = preds_bin.view(-1)
            masks_flat = masks.view(-1)

            intersection = (preds_flat * masks_flat).sum().item()
            pred_sum = preds_flat.sum().item()
            mask_sum = masks_flat.sum().item()

            intersection_sum += intersection
            union_sum += pred_sum + mask_sum

    val_loss = running_loss / len(loader.dataset)

    # Global Dice Formula: 2 * |X n Y| / (|X| + |Y|)
    smooth = 1e-6
    global_dice = (2.0 * intersection_sum) / (union_sum + smooth)

    return val_loss, global_dice


def run_training():
    """
    Main execution function for the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Device: {device}")

    # 2. Load Metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)

    # Debugging: Sample dataset if configured
    if Config.DEBUG_SAMPLE_SIZE is not None:
        print(f"DEBUG MODE: Sampling {Config.DEBUG_SAMPLE_SIZE} records.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # 3. Datasets & Loaders
    train_dataset = ContrailDataset(train_df, split="train")
    val_dataset = ContrailDataset(val_df, split="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Model, Loss, Optimizer
    model = MobileNetUNet(in_channels=Config.IN_CHANNELS, num_classes=1)
    model.to(device)

    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # 5. Training Loop
    best_dice = -1.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    patience_counter = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, threshold=Config.THRESHOLD
        )

        # Step Scheduler
        scheduler.step()

        epoch_duration = time.time() - epoch_start

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Dice: {val_dice}"
        )

        # Checkpointing & Early Stopping
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s. Best Validation Dice: {best_dice}")

    return best_model_path
