import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.dataset import ContrailDataset, get_train_transform, get_valid_transform
from library.model import DeformableResNetUNet
from library.loss import HybridLoss
from library.utils import seed_everything


def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, masks in dataloader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, device, threshold=0.5):
    """
    Evaluates the model on the validation set using Global Dice Coefficient.
    """
    model.eval()

    # Accumulators for Global Dice
    total_intersection = 0.0
    total_union = 0.0
    epsilon = 1e-6

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                probs = torch.sigmoid(outputs)

            # Binarize predictions
            preds = (probs > threshold).float()

            # Flatten for calculation
            preds = preds.view(-1)
            masks = masks.view(-1)

            intersection = (preds * masks).sum().item()
            union = preds.sum().item() + masks.sum().item()

            total_intersection += intersection
            total_union += union

    # Compute Global Dice
    global_dice = (2.0 * total_intersection + epsilon) / (total_union + epsilon)
    return global_dice


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    debug=Config.DEBUG,
    patience=5,
):
    """
    Main function to orchestrate the training process.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Preparation
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    valid_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)

    # Initialize Datasets
    train_dataset = ContrailDataset(
        train_df, transform=get_train_transform(), debug=debug
    )
    valid_dataset = ContrailDataset(
        valid_df, transform=get_valid_transform(), debug=debug
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = DeformableResNetUNet(
        n_channels=Config.N_CHANNELS, n_classes=1, pretrained=True
    ).to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    criterion = HybridLoss(
        bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT
    ).to(device)

    scaler = GradScaler()

    # 5. Training Loop
    best_dice = 0.0
    epochs_no_improve = 0
    start_time = time.time()

    print("Starting training...")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )

        # Validate
        val_dice = validate(model, valid_loader, device, threshold=Config.THRESHOLD)

        # Update Scheduler
        scheduler.step()

        epoch_duration = time.time() - epoch_start

        # Logging
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Time: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Global Dice: {val_dice}"
        )

        # Checkpointing & Early Stopping
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {epochs_no_improve} epochs with no improvement."
            )
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s. Best Validation Dice: {best_dice}")
