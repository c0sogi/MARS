import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.utils import compute_intersection_union, calculate_global_dice

# =========================================================================
# Loss Function
# =========================================================================


class DiceBCELoss(nn.Module):
    """
    Mixed loss combining Binary Cross Entropy (BCE) and Dice Loss.
    Loss = 0.5 * BCE + 0.5 * DiceLoss
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # BCE Loss (inputs are logits)
        bce = self.bce_loss(inputs, targets)

        # Dice Loss
        # Apply sigmoid to convert logits to probabilities
        inputs_soft = torch.sigmoid(inputs)

        # Flatten
        inputs_flat = inputs_soft.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice = (2.0 * intersection + 1e-6) / (
            inputs_flat.sum() + targets_flat.sum() + 1e-6
        )

        dice_loss = 1 - dice

        # Combined Loss
        return 0.5 * bce + 0.5 * dice_loss


# =========================================================================
# Training Step
# =========================================================================


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()

        # Gradient clipping (optional but recommended for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        # Optimization step
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


# =========================================================================
# Validation Step
# =========================================================================


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Global Accumulation.
    """
    model.eval()
    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Forward pass
            outputs = model(images)

            # Apply sigmoid and threshold
            preds = torch.sigmoid(outputs)
            preds = (preds > Config.MASK_THRESHOLD).float()

            # Accumulate stats for Global Dice
            # We compute intersection and union for the batch
            inter, union = compute_intersection_union(preds, masks)
            total_intersection += inter
            total_union += union

    # Calculate global dice
    dice_score = calculate_global_dice(total_intersection, total_union)
    return dice_score


# =========================================================================
# Main Training Loop
# =========================================================================


def run_training(
    model,
    dataloaders,
    optimizer,
    scheduler,
    num_epochs=Config.EPOCHS,
    device=Config.DEVICE,
    patience=5,
):
    """
    Runs the full training loop with early stopping.
    """
    criterion = DiceBCELoss()
    best_dice = 0.0
    epochs_no_improve = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {num_epochs} epochs on {device}...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, dataloaders["train"], optimizer, criterion, device
        )

        # Validate
        val_dice = validate(model, dataloaders["val"], device)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - start_time

        # Logging
        print(
            f"Epoch {epoch+1}/{num_epochs} | Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss:.6f} | Val Dice: {val_dice}"
        )

        # Save Best Model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved! Dice: {best_dice}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early Stopping
        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {patience} epochs with no improvement."
            )
            break

    print(f"Training complete. Best Val Dice: {best_dice}")
    return best_model_path
