import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import set_seed, AverageMeter
from library.dataset import ContrailDataset, get_transforms
from library.model import StripPoolingResNet18UNet
from library.loss import HybridLoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set using Global Dice Coefficient.
    """
    model.eval()
    losses = AverageMeter()

    # Global Dice accumulators
    total_intersection = 0.0
    total_union = 0.0

    # Threshold for binary prediction
    threshold = Config.THRESHOLD
    smooth = 1e-6

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            losses.update(loss.item(), images.size(0))

            # Apply sigmoid
            probs = torch.sigmoid(outputs)

            # Binarize predictions
            preds = (probs > threshold).float()

            # Flatten for global calculation
            preds = preds.view(-1)
            targets = masks.view(-1)

            # Accumulate stats
            intersection = (preds * targets).sum().item()
            pred_sum = preds.sum().item()
            target_sum = targets.sum().item()

            total_intersection += intersection
            total_union += pred_sum + target_sum

    # Compute Global Dice
    global_dice = (2.0 * total_intersection + smooth) / (total_union + smooth)

    return global_dice, losses.avg


def run_training(debug=False):
    """
    Orchestrates the training pipeline.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Initialize Model
    model = StripPoolingResNet18UNet(in_channels=Config.IN_CHANNELS, pretrained=True)
    model = model.to(device)

    # Initialize Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Initialize Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Initialize Loss
    criterion = HybridLoss(bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT)

    # Initialize Datasets and Loaders
    train_dataset = ContrailDataset(
        split="train", transform=get_transforms("train", Config), debug=debug
    )
    val_dataset = ContrailDataset(
        split="validation", transform=get_transforms("validation", Config), debug=debug
    )

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
        drop_last=False,
    )

    print(f"Starting training for {Config.EPOCHS} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    best_dice = 0.0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(Config.BEST_MODEL_PATH), exist_ok=True)

    for epoch in range(Config.EPOCHS):
        # Training Step
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation Step
        val_dice, val_loss = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Global Dice: {val_dice}"
        )

        # Save Best Model
        if val_dice > best_dice:
            print(
                f"Global Dice improved from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    print(f"Training completed. Best Global Dice: {best_dice}")
