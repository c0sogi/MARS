import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from library.config import Config
from library.dataset import get_dataloaders
from library.model import get_artwork_model
from library.loss import AsymmetricLoss
from library.utils import seed_everything, calculate_micro_f1, EarlyStopping


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    """
    Executes one epoch of training.

    Args:
        model: The PyTorch model.
        loader: The training DataLoader.
        criterion: The loss function.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        device: The device to train on.
        scaler: The GradScaler for AMP.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        loader: The validation DataLoader.
        criterion: The loss function.
        device: The device to evaluate on.

    Returns:
        tuple: (Average validation loss, Micro F1 score)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to logits to get probabilities
            preds = torch.sigmoid(outputs)

            # Store on CPU to avoid GPU OOM during metric calculation
            all_preds.append(preds.detach().cpu())
            all_targets.append(targets.detach().cpu())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate Micro F1 using the base threshold from config
    # Threshold calibration is typically done post-training or during inference
    val_f1 = calculate_micro_f1(all_preds, all_targets, threshold=Config.base_threshold)

    return epoch_loss, val_f1


def run_training():
    """
    Orchestrates the entire training pipeline.
    """
    # 1. Initialization
    seed_everything(Config.seed)
    os.makedirs(Config.working_dir, exist_ok=True)
    device = Config.device

    print(f"Initializing training on device: {device}")

    # 2. Data Loading
    # load_cached_data=True enables the use of parquet caching for speed
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Model Setup
    model = get_artwork_model(
        num_classes=Config.num_classes, pretrained=Config.pretrained
    )
    model = model.to(device)

    # 4. Training Components
    criterion = AsymmetricLoss(
        gamma_neg=Config.asl_gamma_neg,
        gamma_pos=Config.asl_gamma_pos,
        clip=Config.asl_clip,
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )

    # OneCycleLR requires the number of steps per epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.lr,
        epochs=Config.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    scaler = GradScaler()

    # 5. Early Stopping
    # Monitors 'max' mode for F1 score
    early_stopping = EarlyStopping(
        patience=5, mode="max", verbose=True, path=Config.model_save_path
    )

    # 6. Training Loop
    print(f"Starting training for {Config.epochs} epochs...")

    for epoch in range(Config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )

        # Validate
        val_loss, val_f1 = validate(model, val_loader, criterion, device)

        elapsed_time = time.time() - start_time

        # Log metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.epochs} "
            f"[Time: {elapsed_time:.2f}s] "
            f"Train Loss: {train_loss} "
            f"Val Loss: {val_loss} "
            f"Val F1: {val_f1}"
        )

        # Check Early Stopping (saves model if improved)
        early_stopping(val_f1, model, optimizer, scheduler, epoch)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation F1: {early_stopping.best_score}")
    return early_stopping.best_score
