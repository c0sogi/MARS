import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from library.config import Config
from library.dataset import ContrailDataset
from library.model import ResnetUNet
from library.loss import GlobalBatchDiceLoss
from library.utils import dice_score_batch


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Executes one epoch of training.
    """
    model.train()

    running_loss = 0.0

    for i, (images, masks, _) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)

    return avg_loss


def validate(model, dataloader, criterion, device, threshold=0.5):
    """
    Evaluates the model on the validation set.
    Computes the Global Dice Coefficient over the entire dataset.
    """
    model.eval()

    running_loss = 0.0

    # Accumulators for Global Dice Calculation
    # Dice = 2 * (Intersection) / (Cardinality)
    # Summed over the entire dataset
    total_intersection = 0.0
    total_cardinality = 0.0

    with torch.no_grad():
        for images, masks, _ in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            # Compute validation loss for monitoring
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Get predictions
            pred_mask = outputs  # Shape [B, 1, H, W]

            # Binarize for metric calculation
            pred_binary = (pred_mask > threshold).float()

            # Flatten for global accumulation
            pred_flat = pred_binary.view(-1)
            target_flat = masks.view(-1)

            intersection = (pred_flat * target_flat).sum().item()
            cardinality = (pred_flat.sum() + target_flat.sum()).item()

            total_intersection += intersection
            total_cardinality += cardinality

    avg_loss = running_loss / len(dataloader)

    # Compute Global Dice
    epsilon = 1e-6
    global_dice = (2.0 * total_intersection) / (total_cardinality + epsilon)

    return avg_loss, global_dice


def train_model(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    max_samples=None,
    num_workers=Config.NUM_WORKERS,
    patience=5,
):
    """
    Main training pipeline.
    """
    # Set seeds for reproducibility
    Config.set_seed(Config.SEED)

    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Prepare Data
    print("Initializing Datasets...")
    train_dataset = ContrailDataset(split="train", max_samples=max_samples)
    val_dataset = ContrailDataset(split="validation", max_samples=max_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Prepare Model, Loss, Optimizer
    print("Initializing Model...")
    model = ResnetUNet(in_channels=Config.IN_CHANNELS, pretrained=True)
    model = model.to(device)

    criterion = GlobalBatchDiceLoss()

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # 3. Training Loop
    best_dice = -1.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, threshold=Config.THRESHOLD
        )

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print Metrics
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  LR: {current_lr:.9f}")
        print(f"  Train Loss: {train_loss:.9f}")
        print(f"  Val Loss:   {val_loss:.9f}")
        print(f"  Val Global Dice: {val_dice:.9f}")

        # Checkpointing & Early Stopping
        if val_dice > best_dice:
            print(
                f"  New Best Dice! ({best_dice:.9f} -> {val_dice:.9f}). Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Global Dice: {best_dice:.9f}")
    print(f"Best model saved to: {best_model_path}")
