import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.dataset import HuBMAPDataset
from library.model import StainNet
from library.loss import DeepSupervisionLoss


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The StainNet model.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        loss_fn (nn.Module): Loss function (DeepSupervisionLoss).
        device (str): Device to run training on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks in loader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        batch_size = images.size(0)

        optimizer.zero_grad()

        # Forward pass
        # Output is a list of tensors if deep_supervision is True
        outputs = model(images)

        # Compute loss
        loss = loss_fn(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, loss_fn, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The StainNet model.
        loader (DataLoader): Validation data loader.
        loss_fn (nn.Module): Loss function.
        device (str): Device to run evaluation on.

    Returns:
        dict: Dictionary containing 'loss' and 'dice'.
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    dataset_size = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            batch_size = images.size(0)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            running_loss += loss.item() * batch_size

            # For Dice calculation, we use the primary output (index 0)
            # If deep supervision is on, outputs is a list.
            if isinstance(outputs, (list, tuple)):
                main_pred = outputs[0]
            else:
                main_pred = outputs

            # Apply sigmoid and threshold
            probs = torch.sigmoid(main_pred)
            preds = (probs > Config.THRESHOLD).float()

            # Calculate Dice for this batch
            # Flatten
            preds_flat = preds.view(batch_size, -1)
            masks_flat = masks.view(batch_size, -1)

            intersection = (preds_flat * masks_flat).sum(dim=1)
            union = preds_flat.sum(dim=1) + masks_flat.sum(dim=1)

            # Smooth dice to avoid division by zero
            smooth = 1e-7
            dice = (2.0 * intersection + smooth) / (union + smooth)
            running_dice += dice.sum().item()

            dataset_size += batch_size

    avg_loss = running_loss / dataset_size
    avg_dice = running_dice / dataset_size

    return {"loss": avg_loss, "dice": avg_dice}


def train_model(load_cached_data=True):
    """
    Main training loop.

    Args:
        load_cached_data (bool): Whether to use cached dataset files.
    """
    Config.setup()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Initialize Datasets and Loaders
    print("Initializing datasets...")
    train_dataset = HuBMAPDataset(mode="train", load_cached_data=load_cached_data)
    val_dataset = HuBMAPDataset(mode="validation", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 2. Initialize Model
    print("Initializing StainNet model...")
    model = StainNet()
    model.to(device)

    # 3. Setup Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    loss_fn = DeepSupervisionLoss()

    # 4. Training Loop
    best_dice = 0.0
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr:.2e}")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)
        val_loss = val_metrics["loss"]
        val_dice = val_metrics["dice"]

        # Step Scheduler
        scheduler.step()

        # Print metrics
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val Dice:   {val_dice}")  # Full precision as requested

        # Checkpoint & Early Stopping
        if val_dice > best_dice:
            print(
                f"  Validation Dice improved from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Dice: {best_dice}")
