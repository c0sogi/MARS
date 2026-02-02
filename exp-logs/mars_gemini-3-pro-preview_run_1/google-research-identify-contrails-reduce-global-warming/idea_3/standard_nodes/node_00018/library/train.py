import os
import torch
import torch.optim as optim
import numpy as np
import time
from library.config import Config
from library.utils import set_seed, dice_coefficient
from library.dataset import get_dataloaders
from library.model import ContrailUNet
from library.loss import ContrailLoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device, threshold=0.5):
    """
    Performs validation loop. Returns average loss and global Dice coefficient.
    """
    model.eval()
    running_loss = 0.0

    # For global Dice, we can accumulate intersection and union,
    # or average batch-wise Dice. The metric definition is Global Dice.
    # Formula: 2 * |X n Y| / (|X| + |Y|) over the entire set.

    intersection_sum = 0.0
    union_sum = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            # Calculate Dice components
            preds = torch.sigmoid(outputs)
            preds_bin = (preds > threshold).float()

            # Flatten for calculation
            preds_flat = preds_bin.view(-1)
            masks_flat = masks.view(-1)

            intersection_sum += (preds_flat * masks_flat).sum().item()
            union_sum += preds_flat.sum().item() + masks_flat.sum().item()

    val_loss = running_loss / len(loader.dataset)

    # Compute Global Dice
    smooth = 1e-6
    global_dice = (2.0 * intersection_sum + smooth) / (union_sum + smooth)

    return val_loss, global_dice


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    num_workers=Config.NUM_WORKERS,
    debug=Config.DEBUG,
    debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    device=Config.DEVICE,
):
    """
    Main function to train the model.
    """
    # 1. Setup
    set_seed(Config.SEED)
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)

    print(f"Starting training on device: {device}")

    # 2. DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        debug=debug,
        debug_sample_size=debug_sample_size,
    )

    # 3. Model, Loss, Optimizer
    model = ContrailUNet().to(device)

    criterion = ContrailLoss().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # 4. Training Loop
    best_dice = 0.0
    early_stopping_patience = 5
    epochs_no_improve = 0
    best_model_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, threshold=Config.THRESHOLD
        )

        # Scheduler Step
        scheduler.step(val_loss)

        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Dice: {val_dice:.10f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.0f}s"
        )

        # Checkpoint
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice:.6f} to {val_dice:.6f}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early Stopping
        if epochs_no_improve >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epochs_no_improve} epochs with no improvement."
            )
            break

    print(f"Training complete. Best Validation Dice: {best_dice:.10f}")
    return best_dice
