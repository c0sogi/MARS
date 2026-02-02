import os
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import MixupCutmix, calculate_f1_score, ModelEMA


def train_one_epoch(
    model, optimizer, data_loader, device, epoch, mixup_fn, model_ema=None
):
    """
    Trains the model for one epoch.
    """
    model.train()

    # Loss function handling soft targets from MixUp/CutMix
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = targets.to(device)

        # Apply MixUp / CutMix
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update EMA
        if model_ema is not None:
            model_ema.update(model)

        # Statistics
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, data_loader, device):
    """
    Validates the model using Test Time Augmentation (TTA).
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    # TTA Configuration
    use_tta = Config.USE_TTA

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # 1. Original Forward Pass
            logits = model(images)

            if use_tta:
                # 2. Horizontal Flip (dim 3)
                images_hflip = torch.flip(images, dims=[3])
                logits_hflip = model(images_hflip)

                # 3. Vertical Flip (dim 2)
                images_vflip = torch.flip(images, dims=[2])
                logits_vflip = model(images_vflip)

                # Average Logits
                logits = (logits + logits_hflip + logits_vflip) / 3.0

            # Compute Loss
            loss = criterion(logits, targets)

            # Apply Sigmoid for metrics
            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            all_preds.append(probs.cpu())
            all_targets.append(targets.cpu())

    epoch_loss = running_loss / dataset_size

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    epoch_f1 = calculate_f1_score(all_targets, all_preds, threshold=Config.THRESHOLD)

    return epoch_loss, epoch_f1


def train_model(
    model, train_loader, val_loader, optimizer, scheduler, device, epochs, patience=5
):
    """
    Main training loop with Early Stopping and EMA handling.
    """
    # Initialize MixUp/CutMix
    mixup_fn = MixupCutmix(
        mixup_alpha=Config.MIXUP_ALPHA,
        cutmix_alpha=Config.CUTMIX_ALPHA,
        prob=Config.MIXUP_PROB,
        switch_prob=Config.SWITCH_PROB,
    )

    # Initialize EMA
    model_ema = None
    if Config.USE_EMA:
        model_ema = ModelEMA(model, decay=Config.EMA_DECAY, device=device)

    best_f1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch, mixup_fn, model_ema
        )

        # Step Scheduler
        if scheduler is not None:
            scheduler.step()

        # Validate
        # Use EMA model for validation if available
        val_model = model_ema.ema if model_ema else model
        val_loss, val_f1 = validate(val_model, val_loader, device)

        # Print Metrics (Full Precision)
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val F1: {val_f1}")

        # Early Stopping & Checkpointing
        if val_f1 > best_f1:
            print(f"Validation F1 improved from {best_f1} to {val_f1}. Saving model...")
            best_f1 = val_f1
            patience_counter = 0

            # Save the best model state
            # If using EMA, save the EMA weights as the primary model for inference
            state_dict = model_ema.ema.state_dict() if model_ema else model.state_dict()
            torch.save(state_dict, best_model_path)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation F1: {best_f1}")
    return best_f1
