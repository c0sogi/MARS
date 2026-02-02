import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import provided library modules
from library.config import Config
from library.utils import set_seed, metric_global_dice
from library.loss import HybridBatchDiceLoss
from library.model import ContextEnhancedUNet
from library.dataset import ContrailsDataset


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # Calculate loss
        loss = criterion(logits, masks)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device, tta=False):
    """
    Evaluates the model on the validation set.
    Computes Global Dice Coefficient and Validation Loss.
    Args:
        tta (bool): If True, applies Test Time Augmentation (HFlip, VFlip, Rot180).
    """
    model.eval()
    running_loss = 0.0

    # Accumulators for Global Dice calculation
    # Dice = 2 * |X n Y| / (|X| + |Y|)
    total_intersection = 0.0
    total_union = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward pass
            logits = model(images)

            if tta:
                # TTA: Original + HFlip + VFlip + Rot180
                probs_1 = torch.sigmoid(logits)

                # HFlip
                logits_h = model(torch.flip(images, dims=[3]))
                probs_h = torch.flip(torch.sigmoid(logits_h), dims=[3])

                # VFlip
                logits_v = model(torch.flip(images, dims=[2]))
                probs_v = torch.flip(torch.sigmoid(logits_v), dims=[2])

                # Rot180 (H+V)
                logits_hv = model(torch.flip(images, dims=[2, 3]))
                probs_hv = torch.flip(torch.sigmoid(logits_hv), dims=[2, 3])

                probs = (probs_1 + probs_h + probs_v + probs_hv) / 4.0
            else:
                probs = torch.sigmoid(logits)

            # Compute loss on original logits (approximate for TTA case)
            loss = criterion(logits, masks)
            running_loss += loss.item() * images.size(0)

            # Generate predictions
            preds = (probs > Config.THRESHOLD).float()

            # Flatten for metric calculation
            preds_flat = preds.view(-1)
            masks_flat = masks.view(-1)

            # Accumulate intersection and union
            intersection = (preds_flat * masks_flat).sum().item()
            union = preds_flat.sum().item() + masks_flat.sum().item()

            total_intersection += intersection
            total_union += union

    val_loss = running_loss / len(loader.dataset)

    # Compute Global Dice
    # Handle edge case where both prediction and ground truth are empty (union=0)
    if total_union == 0:
        global_dice = 1.0
    else:
        global_dice = (2.0 * total_intersection) / total_union

    return val_loss, global_dice


def train_model(debug=Config.DEBUG):
    """
    Main training loop with Early Stopping and Checkpointing.

    Args:
        debug (bool): If True, runs on a small subset of data for testing.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)

    # Subset for debugging if requested
    if debug:
        print(f"Debug mode: Subsetting data to {Config.DEBUG_SAMPLE_SIZE} samples.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Create Datasets and Dataloaders
    train_dataset = ContrailsDataset(train_df, train=True)
    val_dataset = ContrailsDataset(val_df, train=True)  # Validation set also has masks

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop last incomplete batch to stabilize BatchNorm/BatchDice
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 4. Initialize Model, Loss, Optimizer
    model = ContextEnhancedUNet().to(device)

    criterion = HybridBatchDiceLoss(
        bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    # 5. Training Loop
    best_dice = -1.0
    patience = 5
    patience_counter = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss:   {val_loss}")
        print(f"  Val Dice:   {val_dice}")
        print(f"  LR:         {current_lr}")
        print(f"  Time:       {time.time() - epoch_start:.2f}s")

        # Checkpointing
        if val_dice > best_dice:
            print(
                f"  [Improvement] Dice increased from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  [No Improvement] Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete in {total_time:.2f}s. Best Global Dice: {best_dice}")
