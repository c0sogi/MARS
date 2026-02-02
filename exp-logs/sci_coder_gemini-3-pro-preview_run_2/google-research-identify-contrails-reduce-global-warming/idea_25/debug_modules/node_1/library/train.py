import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, MetricMonitor, GlobalDiceTracker
from library.dataset import ContrailDataset, get_transforms
from library.model import ConvNeXtUNet
from library.loss import HybridLoss


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network.
        train_loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (str): Device to train on.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_monitor = MetricMonitor()

    # Iterate over batches
    for i, (images, masks) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        loss_monitor.update(loss.item(), images.size(0))

    return loss_monitor.avg


def validate(model, val_loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network.
        val_loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str): Device to evaluate on.

    Returns:
        tuple: (Average Loss, Global Dice Score)
    """
    model.eval()
    loss_monitor = MetricMonitor()
    dice_tracker = GlobalDiceTracker()

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Update metrics
            loss_monitor.update(loss.item(), images.size(0))

            # Apply sigmoid for Dice calculation
            probs = torch.sigmoid(outputs)
            dice_tracker.update(probs, masks, threshold=Config.THRESHOLD)

    return loss_monitor.avg, dice_tracker.compute()


def train_model(debug=False):
    """
    Main function to execute the training pipeline.

    Args:
        debug (bool): If True, runs with a smaller subset of data/epochs for debugging.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Define cache directories
    train_cache_dir = os.path.join(Config.WORKING_DIR, "cache", "train")
    val_cache_dir = os.path.join(Config.WORKING_DIR, "cache", "validation")
    os.makedirs(train_cache_dir, exist_ok=True)
    os.makedirs(val_cache_dir, exist_ok=True)

    print(f"Starting training on device: {device}")
    print(f"Model: {Config.MODEL_NAME}")
    print(f"Backbone: {Config.BACKBONE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading
    train_dataset = ContrailDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        stage="train",
        transform=get_transforms("train"),
        cache_dir=train_cache_dir,
    )

    val_dataset = ContrailDataset(
        metadata_path=Config.VALIDATION_METADATA_PATH,
        stage="validation",
        transform=get_transforms("validation"),
        cache_dir=val_cache_dir,
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

    # 3. Model, Loss, Optimizer
    model = ConvNeXtUNet()
    model.to(device)

    criterion = HybridLoss()

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    # If debug, reduce T_max
    epochs = Config.EPOCHS
    if debug:
        epochs = 2

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.ETA_MIN)

    # 4. Training Loop
    best_dice = 0.0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Early Stopping Parameters
    patience = 10
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{epochs} | Time: {epoch_time:.2f}s | LR: {current_lr:.8f}"
        )
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss:   {val_loss}")
        print(f"  Val Dice:   {val_dice}")

        # Save Best Model
        if val_dice > best_dice:
            print(
                f"  [Improved] Global Dice increased from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  [No Improvement] Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s. Best Global Dice: {best_dice}")
