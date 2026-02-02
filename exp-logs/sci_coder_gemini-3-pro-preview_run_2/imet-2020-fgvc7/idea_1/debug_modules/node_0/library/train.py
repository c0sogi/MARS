import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import set_seed, AverageMeter, calculate_micro_f1
from library.dataset import ArtworkDataset, get_transforms
from library.model import ArtworkClassifier


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): The training data loader.
        optimizer (Optimizer): The optimizer.
        scheduler (LRScheduler): The learning rate scheduler.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to train on.
        scaler (GradScaler): The gradient scaler for mixed precision.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Scales loss and calls backward() to create scaled gradients
        scaler.scale(loss).backward()

        # Unscales gradients and calls optimizer.step()
        scaler.step(optimizer)

        # Updates the scale for next iteration
        scaler.update()

        # Update scheduler (OneCycleLR updates every step)
        if scheduler is not None:
            scheduler.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): The validation data loader.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to evaluate on.

    Returns:
        tuple: (average_loss, micro_f1_score)
    """
    model.eval()
    losses = AverageMeter()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Standard forward pass (autocast optional for inference, but good for speed)
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            # Store logits and targets for F1 calculation
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate metric
    micro_f1 = calculate_micro_f1(
        all_preds, all_targets, threshold=0.5, from_logits=True
    )

    return losses.avg, micro_f1


def fit(
    epochs=Config.epochs,
    batch_size=Config.batch_size,
    learning_rate=Config.learning_rate,
    debug=Config.debug,
    num_workers=Config.num_workers,
    patience=3,
):
    """
    Main training loop.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Peak learning rate.
        debug (bool): If True, runs on a small subset of data.
        num_workers (int): Number of data loading workers.
        patience (int): Early stopping patience.
    """
    # 1. Setup
    set_seed(Config.seed)
    device = torch.device(Config.device)
    print(f"Training on device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = ArtworkDataset(
        metadata_path=Config.TRAIN_METADATA,
        input_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="train", image_size=Config.image_size),
        mode="train",
        num_classes=Config.num_classes,
    )

    val_dataset = ArtworkDataset(
        metadata_path=Config.VAL_METADATA,
        input_dir=Config.INPUT_DIR,
        transform=get_transforms(mode="valid", image_size=Config.image_size),
        mode="valid",
        num_classes=Config.num_classes,
    )

    if debug:
        print("Debug mode: Subsetting data...")
        indices = np.arange(100)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

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

    # 3. Model Initialization
    print(f"Initializing Model: {Config.model_name}")
    model = ArtworkClassifier(
        model_name=Config.model_name, num_classes=Config.num_classes, pretrained=True
    )
    model.to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    # BCEWithLogitsLoss is standard for multi-label classification
    criterion = nn.BCEWithLogitsLoss()

    # OneCycleLR Scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1,
    )

    # Mixed Precision Scaler
    scaler = GradScaler()

    # 5. Training Loop
    best_f1 = 0.0
    early_stopping_counter = 0

    print("Starting Training...")
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler
        )

        # Validate
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch}/{epochs} - "
            f"Time: {elapsed:.2f}s - "
            f"Train Loss: {train_loss:.16f} - "
            f"Val Loss: {val_loss:.16f} - "
            f"Val Micro F1: {val_f1:.16f}"
        )

        # Checkpointing and Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            early_stopping_counter = 0
            print(f"New best F1! Saving model to {Config.MODEL_PATH}")
            torch.save(model.state_dict(), Config.MODEL_PATH)
        else:
            early_stopping_counter += 1
            print(
                f"No improvement. Early stopping counter: {early_stopping_counter}/{patience}"
            )

        if early_stopping_counter >= patience:
            print("Early stopping triggered. Training finished.")
            break

    print(f"Training complete. Best Validation Micro F1: {best_f1:.16f}")
