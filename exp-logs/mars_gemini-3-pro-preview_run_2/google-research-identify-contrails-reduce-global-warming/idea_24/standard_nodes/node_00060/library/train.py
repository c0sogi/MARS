import os
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed
from library.dataset import ContrailDataset, get_transforms
from library.model import ProgressiveConvNeXtUNet
from library.loss import HybridLoss


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = loss_fn(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device, threshold=0.5):
    """
    Evaluates the model on the validation set using the Global Dice Coefficient.

    Global Dice = 2 * |Intersection_All| / (|Pred_All| + |Target_All|)
    """
    model.eval()
    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            # Flatten to compute intersection and cardinality for this batch
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            intersection = (preds_flat * masks_flat).sum().item()
            union = preds_flat.sum().item() + masks_flat.sum().item()

            total_intersection += intersection
            total_union += union

    # Compute Global Dice
    # Add epsilon to avoid division by zero if both sets are empty
    epsilon = 1e-6
    global_dice = (2.0 * total_intersection) / (total_union + epsilon)

    return global_dice


def run_training(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=False):
    """
    Main training loop.

    Args:
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size.
        debug (bool): If True, uses a small subset of data for debugging.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Starting training on device: {device}")
    print(f"Configuration: Epochs={epochs}, Batch Size={batch_size}, Debug={debug}")

    # 2. Data Preparation
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        train_df = train_df.head(100)
        val_df = val_df.head(50)
        epochs = 2

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

    # 3. Model, Optimizer, Loss
    model = ProgressiveConvNeXtUNet().to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.ETA_MIN)

    loss_fn = HybridLoss()

    # 4. Training Loop
    best_dice = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)

        # Validate
        val_dice = validate(model, val_loader, device, threshold=Config.THRESHOLD)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Time: {elapsed:.2f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Global Dice: {val_dice}"
        )

        # Save Best Model
        if val_dice > best_dice:
            print(f"New best model found! (Dice: {best_dice} -> {val_dice})")
            best_dice = val_dice
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"Training complete. Best Validation Global Dice: {best_dice}")
