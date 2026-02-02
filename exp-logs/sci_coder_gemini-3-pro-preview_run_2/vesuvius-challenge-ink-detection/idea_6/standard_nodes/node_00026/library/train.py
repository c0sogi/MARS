import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import fbeta_score, seed_everything
from library.data import get_dataloaders
from library.model import get_model


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: logits (B, 1, H, W)
        # targets: binary mask (B, 1, H, W)

        preds = torch.sigmoid(preds)

        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        logits = outputs.logits

        # Upsample logits to match mask size (SegFormer outputs 1/4 resolution)
        logits = nn.functional.interpolate(
            logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_score = 0.0

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            logits = outputs.logits

            # Upsample
            logits = nn.functional.interpolate(
                logits, size=masks.shape[-2:], mode="bilinear", align_corners=False
            )

            loss = criterion(logits, masks)
            total_loss += loss.item()

            # Calculate F0.5 Score
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)
            score = fbeta_score(probs, masks, threshold=0.5, beta=0.5)
            total_score += score

    return total_loss / len(loader), total_score / len(loader)


def train_model():
    # Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Data
    print("Loading data...")
    train_loader, val_loader, _, _ = get_dataloaders(load_cached_data=True)

    # Model
    print("Initializing model...")
    model = get_model()
    model.to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # Loss
    bce_fn = nn.BCEWithLogitsLoss()
    dice_fn = DiceLoss()

    def criterion(preds, targets):
        bce = bce_fn(preds, targets)
        dice = dice_fn(preds, targets)
        return Config.BCE_WEIGHT * bce + Config.DICE_WEIGHT * dice

    # Training Loop
    best_val_score = 0.0
    # We initialize best_saved_score with the baseline to ensure we only save if we beat it
    # However, to track 'best so far' for logic, we use 0.0, but check threshold before saving.

    epochs_no_improve = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")
    print(f"Baseline Score Threshold: {Config.BASELINE_SCORE_THRESHOLD}")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val F0.5 Score: {val_score}")
        print(f"LR: {current_lr}")

        # Checkpointing logic
        if val_score > best_val_score:
            best_val_score = val_score
            epochs_no_improve = 0

            # Only save if we beat the hard baseline
            if val_score > Config.BASELINE_SCORE_THRESHOLD:
                print(
                    f"New best score {val_score} > threshold {Config.BASELINE_SCORE_THRESHOLD}. Saving model..."
                )
                torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            else:
                print(
                    f"New best score {val_score} but did not beat threshold {Config.BASELINE_SCORE_THRESHOLD}. Model not saved."
                )
        else:
            epochs_no_improve += 1
            print(
                f"No improvement. Patience: {epochs_no_improve}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        # Early Stopping
        if epochs_no_improve >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_val_score}")
