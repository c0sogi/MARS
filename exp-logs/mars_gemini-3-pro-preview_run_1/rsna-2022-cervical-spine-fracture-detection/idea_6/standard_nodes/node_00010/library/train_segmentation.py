import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from library import config, models, datasets, utils

# ====================================================
# UTILS & LOSS
# ====================================================


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W) with class indices
        """
        num_classes = logits.shape[1]

        # Apply Softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        # targets is (B, H, W) -> (B, 1, H, W) -> (B, C, H, W)
        targets_one_hot = (
            F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        )

        # Calculate intersection and union
        intersection = torch.sum(probs * targets_one_hot, dim=(2, 3))
        union = torch.sum(probs, dim=(2, 3)) + torch.sum(targets_one_hot, dim=(2, 3))

        # Dice score per class per batch: (B, C)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Average over batch and classes
        # We can average over all classes, or focus on vertebrae.
        # Standard Dice Loss averages over all classes.
        return 1.0 - torch.mean(dice_score)


# ====================================================
# TRAINING & VALIDATION LOOPS
# ====================================================


def train_one_epoch(model, loader, optimizer, criterion_ce, criterion_dice, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        # Expand 1-channel image to 3-channel for the backbone
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        optimizer.zero_grad()

        logits = model(images)

        loss_ce = criterion_ce(logits, masks)
        loss_dice = criterion_dice(logits, masks)
        loss = 0.5 * loss_ce + 0.5 * loss_dice

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion_ce, criterion_dice, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

            logits = model(images)

            loss_ce = criterion_ce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = 0.5 * loss_ce + 0.5 * loss_dice

            running_loss += loss.item()

            # Calculate Dice Score for monitoring (excluding background if desired, but here we do all)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Simple Dice calculation for non-background classes (1-7)
            # We aggregate counts over the batch
            for cls in range(1, config.NUM_SEG_CLASSES):
                pred_mask = (preds == cls).float()
                true_mask = (masks == cls).float()

                intersection = (pred_mask * true_mask).sum()
                union = pred_mask.sum() + true_mask.sum()

                score = (2.0 * intersection + 1e-6) / (union + 1e-6)
                dice_scores.append(score.item())

    avg_loss = running_loss / len(loader)
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0

    return avg_loss, avg_dice


# ====================================================
# MAIN RUNNER
# ====================================================


def run_stage1_training(
    epochs=config.STAGE1_CONFIG["epochs"],
    batch_size=config.STAGE1_CONFIG["batch_size"],
    lr=config.STAGE1_CONFIG["lr"],
    patience=3,
):
    set_seed()

    print(f"Starting Stage 1 Training: Multi-Class Anatomical Localizer")
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")

    # 1. Dataset & DataLoader
    train_ds, val_ds = datasets.get_datasets(stage="stage1")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model
    model = models.UNetLocalizer(pretrained=True)
    model = model.to(config.DEVICE)

    # 3. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss()

    # 4. Training Loop
    best_dice = 0.0
    patience_counter = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "stage1_unet.pth")

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_ce, criterion_dice, config.DEVICE
        )
        val_loss, val_dice = validate(
            model, val_loader, criterion_ce, criterion_dice, config.DEVICE
        )

        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Dice (C1-C7): {val_dice}")

        # Scheduler step
        scheduler.step(val_dice)

        # Checkpointing & Early Stopping
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), checkpoint_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Stage 1 Training Completed. Best Val Dice: {best_dice}")
    print(f"Model saved to: {checkpoint_path}")
