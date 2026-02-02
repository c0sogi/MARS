import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    WORKING_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    get_device,
    MODEL_CONFIG,
)
from library.utils import set_seed
from library.dataset import ContrailsDataset
from library.loss import HybridLoss
from library.model import DilatedResNetUNet


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Handles the training loop for a single epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Calculate loss
        loss = criterion(logits, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Computes the Global Dice Coefficient: 2 * |X n Y| / (|X| + |Y|)
    where X is the set of all predicted pixels and Y is the set of all GT pixels.
    """
    model.eval()
    running_loss = 0.0

    # Accumulators for Global Dice
    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

            # Predictions: Sigmoid -> Threshold 0.5
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            # Flatten for calculation
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            intersection = (preds_flat * masks_flat).sum().item()
            pred_sum = preds_flat.sum().item()
            mask_sum = masks_flat.sum().item()

            total_intersection += intersection
            total_union += pred_sum + mask_sum

    val_loss = running_loss / len(loader.dataset)

    # Calculate Global Dice
    # Add small epsilon to avoid division by zero if both sets are empty
    epsilon = 1e-6
    global_dice = (2.0 * total_intersection + epsilon) / (total_union + epsilon)

    return val_loss, global_dice


def train_model(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    debug_size=None,
    patience=10,
):
    """
    Main function to train the Dilated ResNet U-Net model.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate for AdamW.
        debug_size (int, optional): If set, limits dataset size for debugging.
        patience (int): Early stopping patience epochs.
    """
    set_seed()
    device = get_device()

    print(f"Initializing training on device: {device}")
    print(f"Working directory: {WORKING_DIR}")

    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # --- Data Loading ---
    print("Loading datasets...")
    train_dataset = ContrailsDataset(
        metadata_path=TRAIN_METADATA_PATH, split="train", debug_size=debug_size
    )

    val_dataset = ContrailsDataset(
        metadata_path=VAL_METADATA_PATH, split="validation", debug_size=debug_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --- Model Setup ---
    print("Initializing Dilated ResNet18 U-Net (Output Stride 8)...")
    model = DilatedResNetUNet(config=MODEL_CONFIG)
    model = model.to(device)

    # --- Optimization ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Hybrid Loss (BCE + BatchDice)
    criterion = HybridLoss()

    # --- Training Loop ---
    best_dice = 0.0
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{epochs} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Global Dice: {val_dice}"
        )

        # Checkpointing
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    print(f"Training complete. Best Global Dice: {best_dice}")
    print(f"Best model saved to: {best_model_path}")
