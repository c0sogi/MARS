import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.utils import set_seed, fbeta_score, dice_coef
from library.model import HPUnet
from library.data import get_loaders


class BCEDiceLoss(nn.Module):
    """
    Composite loss function: Binary Cross Entropy + Soft Dice Loss.
    BCE handles pixel-wise classification, while Dice optimizes for intersection/overlap.
    """

    def __init__(self, bce_weight=0.5, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1 - bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE Loss
        bce_loss = self.bce(logits, targets)

        # Soft Dice Loss
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice_score

        # Composite
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, F0.5 score, and Dice score.
    """
    model.eval()
    running_loss = 0.0
    running_f05 = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, masks)

            running_loss += loss.item()

            # Apply sigmoid for metrics
            probs = torch.sigmoid(logits)

            # Calculate metrics
            # Note: fbeta_score and dice_coef in utils expect (preds, targets)
            # and handle binarization internally based on threshold.
            batch_f05 = fbeta_score(probs, masks, beta=0.5, threshold=Config.THRESHOLD)
            batch_dice = dice_coef(probs, masks, threshold=Config.THRESHOLD)

            running_f05 += batch_f05
            running_dice += batch_dice

    num_batches = len(loader)
    return (
        running_loss / num_batches,
        running_f05 / num_batches,
        running_dice / num_batches,
    )


def train_model(load_cached_data=True):
    """
    Main training loop.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.

    Returns:
        float: Best validation F0.5 score achieved.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {Config.ENCODER_NAME} (HPUnet)")
    print(f"Input Channels: {Config.IN_CHANNELS}")

    # 2. Data Loading
    print("Loading Metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Debug Mode Handling
    if Config.DEBUG:
        print(f"DEBUG MODE: Subsampling {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
        epochs = Config.DEBUG_EPOCHS
    else:
        epochs = Config.EPOCHS

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    train_loader, val_loader = get_loaders(
        train_df, val_df, load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    model = HPUnet(in_channels=Config.IN_CHANNELS, classes=Config.CLASSES)
    model.to(device)

    # 4. Optimizer, Scheduler, Loss
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Maximize F0.5 score
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    criterion = BCEDiceLoss(bce_weight=0.5)
    scaler = GradScaler()

    # 5. Training Loop
    best_f05 = 0.0

    print("\nStarting Training...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validate
        val_loss, val_f05, val_dice = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step(val_f05)

        # Checkpointing
        saved_msg = ""
        if val_f05 > best_f05:
            best_f05 = val_f05
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            saved_msg = "-> Model Saved!"

        elapsed = time.time() - start_time

        # Logging (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | "
            f"Val F0.5: {val_f05} | Val Dice: {val_dice} {saved_msg}"
        )

    print(f"\nTraining Complete. Best Validation F0.5: {best_f05}")
    return best_f05
