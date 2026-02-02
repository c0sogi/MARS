import os
import time
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, calculate_auc, get_device
from library.data import get_dataloaders
from library.models import get_model


def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Torch device.

    Returns:
        float: Average training loss.
    """
    model.train()
    losses = AverageMeter()

    # Iterate over training data
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        # The model outputs a single logit (B, 1) or (B) depending on squeeze
        # We ensure shape matching for BCEWithLogitsLoss
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_one_epoch(model, val_loader, criterion, device, use_tta=True):
    """
    Validates the model on the validation set.

    Implements Test Time Augmentation (TTA) for metric calculation if enabled.
    Loss is always calculated on the original view for consistency.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): Validation data loader.
        criterion: Loss function.
        device: Torch device.
        use_tta (bool): Whether to use TTA for AUC calculation.

    Returns:
        tuple: (Average validation loss, Validation AUC)
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 1. Calculate Loss on Original View
            outputs_orig = model(images).squeeze(1)
            loss = criterion(outputs_orig, labels)
            losses.update(loss.item(), images.size(0))

            # 2. Calculate Predictions (with TTA if enabled)
            if use_tta:
                # View 1: Original
                prob_1 = torch.sigmoid(outputs_orig)

                # View 2: Horizontal Flip
                images_h = torch.flip(images, [3])
                prob_2 = torch.sigmoid(model(images_h).squeeze(1))

                # View 3: Vertical Flip
                images_v = torch.flip(images, [2])
                prob_3 = torch.sigmoid(model(images_v).squeeze(1))

                # View 4: Both Flips
                images_hv = torch.flip(images, [2, 3])
                prob_4 = torch.sigmoid(model(images_hv).squeeze(1))

                # Average probabilities
                avg_probs = (prob_1 + prob_2 + prob_3 + prob_4) / 4.0
                preds = avg_probs
            else:
                # No TTA
                preds = torch.sigmoid(outputs_orig)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Calculate AUC
    auc = calculate_auc(all_targets, all_preds)

    return losses.avg, auc


def run_fold_training(
    fold_id: int,
    model_name: str,
    num_epochs: int = Config.NUM_EPOCHS,
    patience: int = 10,
    load_cached_data: bool = True,
):
    """
    Orchestrates the training process for a specific fold and model architecture.

    Args:
        fold_id (int): The fold index (0-4).
        model_name (str): Name of the model architecture (timm).
        num_epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        load_cached_data (bool): Whether to use cached data splits.
    """
    device = get_device()
    print(f"Starting training for {model_name} - Fold {fold_id} on {device}")

    # 1. Data Loading
    train_loader, val_loader = get_dataloaders(
        fold_id=fold_id,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
    )

    # 2. Model Initialization
    model = get_model(model_name, pretrained=True)
    model = model.to(device)

    # 3. Optimization Setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0

    # Define save path
    # Clean model name for filename (remove dots usually found in timm names)
    safe_model_name = model_name.split(".")[0]
    save_path = os.path.join(Config.WORK_DIR, f"{safe_model_name}_fold_{fold_id}.pth")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            epoch, model, train_loader, criterion, optimizer, device
        )

        # Validate (using TTA for metric as per strategy)
        val_loss, val_auc = validate_one_epoch(
            model, val_loader, criterion, device, use_tta=True
        )

        elapsed = time.time() - start_time

        # Logging (Full precision)
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC (TTA): {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  [+] Saved best model to {save_path}")
        else:
            patience_counter += 1
            print(f"  [-] No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"Fold {fold_id} finished. Best AUC: {best_auc}")

    # Clean up to free VRAM
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
