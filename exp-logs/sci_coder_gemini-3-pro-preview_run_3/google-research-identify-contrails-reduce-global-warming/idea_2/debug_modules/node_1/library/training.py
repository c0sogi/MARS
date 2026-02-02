import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from library.config import Config
from library.utils import set_seed, dice_coef, average_weights
from library.data import get_loaders
from library.model import ContrailUNet


class DiceBCELoss(nn.Module):
    """
    Hybrid loss function combining Binary Cross Entropy and Dice Loss.
    Useful for segmentation tasks with class imbalance.
    """

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE Loss
        bce = self.bce_loss(logits, targets)

        # Dice Loss
        probs = torch.sigmoid(logits)
        dice_score = dice_coef(probs, targets, smooth=self.smooth)
        dice_loss = 1.0 - dice_score

        # Combined Loss
        loss = self.bce_weight * bce + self.dice_weight * dice_loss
        return loss


class CheckpointManager:
    """
    Manages saving and deleting checkpoints to maintain only the Top-K best models.
    Also handles averaging the weights of the Top-K models.
    """

    def __init__(self, save_dir, top_k=5):
        self.save_dir = save_dir
        self.top_k = top_k
        # List of tuples: (score, epoch, filepath)
        self.checkpoints = []

    def update(self, model, score, epoch):
        """
        Updates the checkpoint list with the current model if it qualifies.
        """
        filename = f"checkpoint_epoch_{epoch}_dice_{score:.6f}.pth"
        filepath = os.path.join(self.save_dir, filename)

        # Save current model temporarily
        torch.save(model.state_dict(), filepath)

        # Add to list
        self.checkpoints.append((score, epoch, filepath))

        # Sort by score descending
        self.checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Keep only Top-K
        if len(self.checkpoints) > self.top_k:
            # Remove the worst checkpoint from the list and disk
            worst_score, worst_epoch, worst_path = self.checkpoints.pop()
            if os.path.exists(worst_path):
                os.remove(worst_path)
                # print(f"Removed checkpoint: {worst_path}")

    def save_average_model(self, output_path):
        """
        Loads the Top-K checkpoints, averages their weights, and saves the result.
        """
        if not self.checkpoints:
            print("No checkpoints to average.")
            return

        print(f"Averaging weights from {len(self.checkpoints)} checkpoints...")
        state_dicts = []
        for score, epoch, path in self.checkpoints:
            try:
                sd = torch.load(path, map_location="cpu")
                state_dicts.append(sd)
                print(f" - Included: Epoch {epoch} (Dice: {score:.6f})")
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if state_dicts:
            avg_sd = average_weights(state_dicts)
            torch.save(avg_sd, output_path)
            print(f"Averaged model saved to {output_path}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    model.eval()

    # Variables for Global Dice calculation
    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Flatten for calculation
            preds = preds.view(-1)
            targets = masks.view(-1)

            intersection = (preds * targets).sum().item()
            union = preds.sum().item() + targets.sum().item()

            total_intersection += intersection
            total_union += union

    # Global Dice: 2 * |X n Y| / (|X| + |Y|)
    # Add epsilon to avoid division by zero
    global_dice = (2.0 * total_intersection) / (total_union + 1e-7)
    return global_dice


def train_model(debug=False):
    """
    Main training loop.
    """
    set_seed(Config.SEED)

    # Initialize Checkpoint Manager
    ckpt_manager = CheckpointManager(
        Config.CHECKPOINT_DIR, top_k=Config.TOP_K_CHECKPOINTS
    )

    # Data Loaders
    train_loader, val_loader, _ = get_loaders(debug=debug, batch_size=Config.BATCH_SIZE)

    # Model
    print(f"Initializing {Config.ENCODER_NAME} U-Net...")
    model = ContrailUNet(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=Config.N_CHANNELS,
        classes=1,
    )
    model = model.to(Config.DEVICE)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # Loss
    criterion = DiceBCELoss()

    print(f"Starting training for {Config.EPOCHS} epochs on {Config.DEVICE}...")

    best_global_dice = 0.0

    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )

        # Validate
        val_dice = validate(model, val_loader, Config.DEVICE)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Global Dice: {val_dice}"
        )

        # Checkpoint Management
        ckpt_manager.update(model, val_dice, epoch)

        if val_dice > best_global_dice:
            best_global_dice = val_dice

    print(f"Training complete. Best Validation Dice: {best_global_dice}")

    # Save Averaged Model
    ckpt_manager.save_average_model(Config.BEST_MODEL_PATH)
