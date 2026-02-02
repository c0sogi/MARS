import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, GlobalDiceTracker, AverageMeter
from library.dataset import ContrailDataset, get_transforms
from library.model import CascadedUNet
from library.loss import DeepSupervisionLoss


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Handles the training of one epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass: Returns (stage1_logits, stage2_logits)
        outputs = model(images)

        # Compute Deep Supervision Loss
        loss, _ = criterion(outputs, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using Global Dice.
    """
    model.eval()
    tracker = GlobalDiceTracker()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # We only care about the final stage output for validation metrics
            _, logits2 = model(images)

            # Convert logits to probabilities
            probs = torch.sigmoid(logits2)

            # Update global dice tracker
            tracker.update(probs, masks)

    return tracker.compute()


def fit(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    debug=False,
):
    """
    Main training loop.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loaders.
        learning_rate (float): Initial learning rate.
        debug (bool): If True, runs on a small subset of data for testing.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 2. Data Preparation
    print("Loading metadata...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} records.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    train_dataset = ContrailDataset(
        train_df, split="train", transform=get_transforms("train")
    )
    val_dataset = ContrailDataset(
        val_df, split="validation", transform=get_transforms("validation")
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Cascaded ResNet18 U-Net...")
    model = CascadedUNet().to(device)

    # 4. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=Config.ETA_MIN
    )

    # Deep Supervision Loss (Stage 1 + Stage 2)
    criterion = DeepSupervisionLoss()

    # 5. Training Loop
    best_dice = 0.0
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_dice = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss} | "
            f"Val Global Dice: {val_dice}"
        )

        # Checkpoint
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training finished. Best Validation Global Dice: {best_dice}")
