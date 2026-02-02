import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WORKING_DIR,
)
from library.utils import set_seed, dice_coef
from library.dataset import get_dataloader
from library.model import UNet, DiceLoss


def train_model(
    debug=False,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    patience=5,
    save_path=MODEL_SAVE_PATH,
):
    """
    Trains the Symmetric U-Net++ model with Early Stopping and Cosine Annealing.

    Args:
        debug (bool): If True, uses a small subset of data.
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        lr (float): Initial learning rate.
        patience (int): Number of epochs to wait for improvement before early stopping.
        save_path (str): Path to save the best model weights.
    """

    # 1. Setup
    set_seed()
    print(f"Initializing training on device: {DEVICE}")

    # Ensure working directory exists for model saving
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 2. Data Loaders
    train_loader = get_dataloader("train", batch_size=batch_size, debug=debug)
    val_loader = get_dataloader("validation", batch_size=batch_size, debug=debug)

    # 3. Model, Optimizer, Scheduler
    model = UNet().to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 4. Loss Functions
    # Combine BCE (pixel-wise classification) and Dice (overlap)
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()

    # 5. Training Loop State
    best_dice = -1.0
    patience_counter = 0

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0
        train_steps = 0

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            logits = model(images)

            # Compute Loss (50% BCE, 50% Dice)
            loss_bce = criterion_bce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = 0.5 * loss_bce + 0.5 * loss_dice

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()
            train_steps += 1

        avg_train_loss = train_loss_accum / train_steps if train_steps > 0 else 0.0

        # --- Validation Phase ---
        model.eval()
        val_loss_accum = 0.0
        val_dice_accum = 0.0
        val_steps = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(images)

                # Validation Loss
                loss_bce = criterion_bce(logits, masks)
                loss_dice = criterion_dice(logits, masks)
                loss = 0.5 * loss_bce + 0.5 * loss_dice
                val_loss_accum += loss.item()

                # Validation Metric (Dice Coefficient)
                # Apply sigmoid and threshold
                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).float()

                batch_dice = dice_coef(preds, masks).item()
                val_dice_accum += batch_dice
                val_steps += 1

        avg_val_loss = val_loss_accum / val_steps if val_steps > 0 else 0.0
        avg_val_dice = val_dice_accum / val_steps if val_steps > 0 else 0.0

        # --- Scheduler Step ---
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # --- Logging ---
        # Printing full precision as requested
        print(f"Epoch {epoch + 1}/{epochs} | LR: {current_lr:.8f}")
        print(f"Train Loss: {avg_train_loss}")
        print(f"Val Loss:   {avg_val_loss}")
        print(f"Val Dice:   {avg_val_dice}")

        # --- Checkpointing & Early Stopping ---
        if avg_val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice} to {avg_val_dice}. Saving model to {save_path}..."
            )
            best_dice = avg_val_dice
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement in Validation Dice. Patience: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        print("-" * 30)

    print(f"Training complete. Best Validation Dice: {best_dice}")
