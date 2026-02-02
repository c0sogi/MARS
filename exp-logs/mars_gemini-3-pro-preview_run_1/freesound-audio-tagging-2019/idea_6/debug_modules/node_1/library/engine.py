import os
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_lwlrap
from library.augmentations import mixup_data, SpecAugment


def train_one_epoch(model, loader, criterion, optimizer, device, spec_augmentor=None):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to run on.
        spec_augmentor (nn.Module, optional): SpecAugment module.

    Returns:
        float: Average training loss.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, batch_data in enumerate(loader):
        images = batch_data["image"].to(device)
        targets = batch_data["target"].to(device)

        # Apply SpecAugment if provided
        if spec_augmentor is not None:
            with torch.no_grad():
                images = spec_augmentor(images)

        # Apply Mixup
        # We use the alpha from Config
        images, targets = mixup_data(
            images, targets, alpha=Config.MIXUP_ALPHA, device=device
        )

        # Forward pass
        logits = model(images)

        # Calculate loss
        loss = criterion(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run on.

    Returns:
        tuple: (Average validation loss, LWLRAP score)
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_data in loader:
            images = batch_data["image"].to(device)
            targets = batch_data["target"].to(device)

            # Forward pass
            logits = model(images)
            loss = criterion(logits, targets)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities for metric calculation
            preds = torch.sigmoid(logits)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metric
    lwlrap = calculate_lwlrap(all_targets, all_preds)

    return losses.avg, lwlrap


def fit(
    model, train_loader, val_loader, optimizer, scheduler, device, epochs=Config.EPOCHS
):
    """
    Main training loop with early stopping and checkpointing.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training loader.
        val_loader (DataLoader): Validation loader.
        optimizer (Optimizer): Optimizer.
        scheduler (LRScheduler): Learning rate scheduler.
        device (torch.device): Device.
        epochs (int): Number of epochs to train.
    """
    criterion = nn.BCEWithLogitsLoss()
    spec_augmentor = SpecAugment().to(device)

    best_score = 0.0
    best_epoch = 0

    # Early stopping settings
    patience = 10
    patience_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, spec_augmentor
        )

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        # Print Metrics (Full precision as requested)
        print(f"Epoch {epoch}/{epochs} | LR: {current_lr}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val LWLRAP: {val_score}")

        # Checkpointing
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            patience_counter = 0
            save_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered. Best score was {best_score} at epoch {best_epoch}."
            )
            break

    print(f"Training complete. Best LWLRAP: {best_score}")
