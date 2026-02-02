import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.model import InkSegFormer
from library.data import get_loaders
from library.utils import dice_coefficient


def set_seed(seed):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Computes 1 - Dice Coefficient (F1 Score).
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Calculate intersection
        intersection = (probs * targets).sum()

        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


def train_model(load_cached_data=True):
    """
    Main training loop for the InkSegFormer model.

    Args:
        load_cached_data (bool): If True, attempts to load pre-processed volumes from cache.
                                 If False, re-processes volumes and updates cache.
    """
    # 1. Set Reproducibility
    set_seed(Config.SEED)

    # 2. Setup Device and Directories
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    save_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Initializing training on device: {device}")

    # 3. Data Loaders
    train_loader, val_loader = get_loaders(load_cached_data=load_cached_data)

    # 4. Model Initialization
    model = InkSegFormer()
    model = model.to(device)

    # 5. Optimization Setup
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # 6. Loss Functions
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    # 7. Training State
    best_val_score = 0.0
    baseline_score = 0.598  # Validation Gating Threshold

    # 8. Epoch Loop
    for epoch in range(Config.EPOCHS):
        # --- Training Step ---
        model.train()
        train_loss_accum = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate combined loss
            loss_bce = criterion_bce(outputs, masks)
            loss_dice = criterion_dice(outputs, masks)
            loss = loss_bce + loss_dice

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation Step ---
        model.eval()
        val_loss_accum = 0.0
        val_score_accum = 0.0

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                # Validation Loss
                l_bce = criterion_bce(outputs, masks)
                l_dice = criterion_dice(outputs, masks)
                val_loss = l_bce + l_dice
                val_loss_accum += val_loss.item()

                # Validation Metric (F0.5 Score)
                # dice_coefficient applies sigmoid internally
                score = dice_coefficient(
                    outputs, masks, threshold=Config.THRESHOLD, beta=0.5
                )
                val_score_accum += score

        avg_val_loss = val_loss_accum / len(val_loader)
        avg_val_score = val_score_accum / len(val_loader)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {avg_train_loss} - Val Loss: {avg_val_loss} - Val F0.5: {avg_val_score}"
        )

        # --- Checkpointing ---
        # Save only if we beat the previous best AND the baseline
        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            if best_val_score > baseline_score:
                torch.save(model.state_dict(), save_path)
                print(
                    f"New best model saved to {save_path} with F0.5: {best_val_score}"
                )
