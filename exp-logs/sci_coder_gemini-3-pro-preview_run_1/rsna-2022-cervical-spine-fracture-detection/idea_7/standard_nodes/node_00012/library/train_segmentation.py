import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.models import SegmentationUNet
from library.data import SegmentationDataset


class DiceLoss(nn.Module):
    """
    Multi-class Dice Loss implementation.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) raw output from model
            targets: (B, H, W) integer class labels
        """
        num_classes = logits.shape[1]
        # Apply Softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, H, W) -> (B, H, W, C) -> (B, C, H, W)
        targets_one_hot = (
            F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        )

        # Calculate intersection and union over spatial dimensions (H, W)
        # Result shape: (B, C)
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        # Dice score per class per batch item
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Average over classes and batch
        return 1.0 - dice_score.mean()


def train_one_epoch(model, loader, optimizer, criterion_ce, criterion_dice, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Model returns: seg_logits, global_context, anatomical_probs
        # We only need seg_logits for Stage 1 training
        logits, _, _ = model(images)

        # Calculate Loss
        loss_ce = criterion_ce(logits, masks)
        loss_dice = criterion_dice(logits, masks)
        loss = loss_ce + loss_dice

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion_ce, criterion_dice, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits, _, _ = model(images)

            loss_ce = criterion_ce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = loss_ce + loss_dice

            running_loss += loss.item()

            # Calculate Dice Score metric (1 - DiceLoss)
            # We use the same logic as the loss but return the score
            dice_val = 1.0 - loss_dice.item()
            running_dice += dice_val

    return running_loss / len(loader), running_dice / len(loader)


def train_stage1(load_cached_data=True):
    """
    Main function to train the Stage 1 Segmentation U-Net.

    Args:
        load_cached_data (bool): Whether to use cached dataset indices/files.
    """
    print("Initializing Stage 1: Segmentation Training...")

    # 1. Setup Device and Config
    device = torch.device(Config.DEVICE)
    Config.setup()

    # 2. Prepare Datasets and Loaders
    train_dataset = SegmentationDataset(
        split="train", load_cached_data=load_cached_data
    )
    val_dataset = SegmentationDataset(split="val", load_cached_data=load_cached_data)

    # If datasets are empty (e.g. debugging with no segmentations found), exit gracefully
    if len(train_dataset) == 0:
        print("No training data found for segmentation. Exiting Stage 1.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.STAGE1_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.STAGE1_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == "cuda" else False,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 3. Model, Optimizer, Loss
    model = SegmentationUNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.STAGE1_LR, weight_decay=Config.STAGE1_WEIGHT_DECAY
    )

    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss()

    # 4. Training Loop
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "stage1_unet.pth")

    print(f"Starting training for {Config.STAGE1_EPOCHS} epochs...")

    for epoch in range(Config.STAGE1_EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion_ce, criterion_dice, device
        )

        # Validate
        val_loss, val_dice = validate(
            model, val_loader, criterion_ce, criterion_dice, device
        )

        print(
            f"Epoch {epoch+1}/{Config.STAGE1_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Dice Score: {val_dice:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  New best model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Stage 1 Training Completed.")
