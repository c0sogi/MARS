import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torch.cuda.amp import GradScaler, autocast

# Import from local library
from library.config import Config
from library.utils import set_seed, dice_coeff
from library.dataset import HuBMAPDataset
from library.model import LinkNetResNet34


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    Useful for segmentation tasks with class imbalance.
    """

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, pred, target):
        # BCE Loss (expects logits)
        bce = self.bce_loss(pred, target)

        # Dice Loss (expects probabilities)
        pred_probs = torch.sigmoid(pred)

        # Flatten for Dice calculation
        # Ensure float32 for stability, especially with AMP
        pred_flat = pred_probs.view(-1).float()
        target_flat = target.view(-1).float()

        intersection = (pred_flat * target_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score

        # Combined Loss
        return self.bce_weight * bce + self.dice_weight * dice_loss


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    """
    Trains the model for one epoch using Mixed Precision.
    """
    model.train()
    running_loss = 0.0

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        # Mixed Precision Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device, threshold=0.5):
    """
    Evaluates the model on the validation set using global accumulation.
    Cite solution_lesson_node_00002: Avoids averaging tile metrics.
    """
    model.eval()
    running_loss = 0.0

    # Global accumulators for Dice calculation
    intersection_sum = 0.0
    pred_sum = 0.0
    target_sum = 0.0
    smooth = 1e-6

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)

            # Mixed precision for validation as well
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid and threshold
            preds_probs = torch.sigmoid(outputs)
            preds = (preds_probs > threshold).float()

            # Flatten for global accumulation
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            intersection_sum += (preds_flat * masks_flat).sum().item()
            pred_sum += preds_flat.sum().item()
            target_sum += masks_flat.sum().item()

    epoch_loss = running_loss / len(loader.dataset)

    # Compute global Dice
    epoch_dice = (2.0 * intersection_sum + smooth) / (pred_sum + target_sum + smooth)

    return epoch_loss, epoch_dice


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    load_cached_data=True,
):
    """
    Main function to train the model.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay for optimizer.
        load_cached_data (bool): Whether to load cached dataset files.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on device: {device}")

    # --- Load Metadata ---
    if not os.path.exists(Config.TRAIN_METADATA_PATH) or not os.path.exists(
        Config.VAL_METADATA_PATH
    ):
        raise FileNotFoundError(
            "Metadata files not found. Please ensure metadata generation script has run."
        )

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Debug mode: subset data
    if Config.DEBUG:
        print(f"Debug mode enabled. Using {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # --- Dataset & Dataloader ---
    train_dataset = HuBMAPDataset(
        train_df, mode="train", load_cached_data=load_cached_data
    )
    # Use 0.0 overlap for validation to ensure global metric correctness (sum of tiles = full image)
    val_dataset = HuBMAPDataset(
        val_df, mode="val", load_cached_data=load_cached_data, overlap=0.0
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train tiles: {len(train_dataset)}, Val tiles: {len(val_dataset)}")

    # --- Model, Loss, Optimizer ---
    model = LinkNetResNet34(in_channels=Config.IN_CHANNELS, classes=Config.CLASSES)
    model = model.to(device)

    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Initialize Scaler for AMP
    scaler = GradScaler()

    # --- Training Loop ---
    best_dice = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, threshold=Config.THRESHOLD
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Dice: {val_dice:.4f} - "
            f"Time: {elapsed:.2f}s"
        )

        # --- Checkpoint & Early Stopping ---
        if val_dice > best_dice + Config.EARLY_STOPPING_MIN_DELTA:
            best_dice = val_dice
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT_PATH)
            print(f"New best model saved with Dice: {best_dice:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Dice: {best_dice:.4f}")
    return best_dice
