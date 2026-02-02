import time
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_roc_auc, save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (str): Device to run training on.
        epoch (int): Current epoch number.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.train()

    losses = AverageMeter()

    # Store predictions and targets for AUC calculation
    all_targets = []
    all_preds = []

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device).float().view(-1, 1)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), images.size(0))

        # Store for AUC
        # Apply sigmoid to convert logits to probabilities for AUC calculation
        preds = torch.sigmoid(outputs).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(targets_np)

    # Calculate epoch metrics
    epoch_loss = losses.avg
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model on the validation set.

    Args:
        model (nn.Module): The model to validate.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str): Device to run validation on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()

    losses = AverageMeter()

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).float().view(-1, 1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Update metrics
            losses.update(loss.item(), images.size(0))

            # Store for AUC
            preds = torch.sigmoid(outputs).cpu().numpy()
            targets_np = targets.cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(targets_np)

    # Calculate epoch metrics
    epoch_loss = losses.avg
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    fold_idx,
    model_name_suffix="",
):
    """
    Orchestrates the training loop, including early stopping and checkpointing.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (str): Device to use.
        fold_idx (int): Current fold index.
        model_name_suffix (str): Suffix for the model filename (e.g., architecture name).

    Returns:
        float: The best validation score achieved (based on EARLY_STOPPING_MONITOR).
    """
    criterion = nn.BCEWithLogitsLoss()

    best_score = float("inf") if Config.EARLY_STOPPING_MODE == "min" else -float("inf")
    early_stop_counter = 0

    print(f"Starting training for Fold {fold_idx} on device {device}...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        # Scheduler Step
        if scheduler:
            scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics (Full precision)
        print(f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {elapsed}s")
        print(f"Train Loss: {train_loss} | Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss} | Val AUC: {val_auc}")

        # Early Stopping Logic
        current_score = (
            val_loss if Config.EARLY_STOPPING_MONITOR == "val_loss" else val_auc
        )

        improved = False
        if Config.EARLY_STOPPING_MODE == "min":
            if current_score < best_score:
                improved = True
        else:
            if current_score > best_score:
                improved = True

        if improved:
            best_score = current_score
            early_stop_counter = 0

            # Save Checkpoint
            # Construct filename: e.g., resnet34_fold_0.pth
            filename = f"{model_name_suffix}_fold_{fold_idx}.pth"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                score=best_score,
                path=Config.WORKING_DIR,
                filename=filename,
            )
            print(f"Score improved. Model saved to {Config.WORKING_DIR}/{filename}")
        else:
            early_stop_counter += 1
            print(
                f"No improvement. Early stopping counter: {early_stop_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered. Training finished.")
            break

    return best_score
