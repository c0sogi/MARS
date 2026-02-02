import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything, fbeta_score
from library.dataset import InkDataset
from library.model import build_model


class BCETverskyLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Tversky Loss.
    Tversky Loss is configured to optimize for F0.5 score by penalizing False Positives more.
    Alpha=0.7, Beta=0.3 corresponds to emphasizing Precision.
    """

    def __init__(self, alpha=0.7, beta=0.3, bce_weight=0.5, smooth=1e-6):
        super(BCETverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_weight = bce_weight
        self.tversky_weight = 1.0 - bce_weight
        self.smooth = smooth
        self.bce_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, preds, targets, mask):
        # preds: (B, 1, H, W) logits
        # targets: (B, 1, H, W) binary
        # mask: (B, 1, H, W) valid pixels (1=valid, 0=padding)

        # 1. BCE Loss (Masked)
        bce_loss_pixel = self.bce_fn(preds, targets)
        # Apply mask: multiply loss by mask, sum, divide by number of valid pixels
        valid_pixel_count = mask.sum() + self.smooth
        bce_loss = (bce_loss_pixel * mask).sum() / valid_pixel_count

        # 2. Tversky Loss (Masked)
        preds_sigmoid = torch.sigmoid(preds)

        # Flatten tensors
        preds_flat = preds_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        mask_flat = mask.view(-1)

        # Filter by mask
        preds_flat = preds_flat * mask_flat
        targets_flat = targets_flat * mask_flat

        # Calculate TP, FP, FN
        tp = (preds_flat * targets_flat).sum()
        fp = (preds_flat * (1 - targets_flat)).sum()
        fn = ((1 - preds_flat) * targets_flat).sum()

        # Tversky Index
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        tversky_loss = 1.0 - tversky_index

        # Combined
        total_loss = self.bce_weight * bce_loss + self.tversky_weight * tversky_loss
        return total_loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, labels, masks, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels, masks, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels, masks)
            running_loss += loss.item()

            # For F0.5 score, we need to consider the mask.
            # We only evaluate on valid pixels.
            # Flatten and mask
            preds_prob = torch.sigmoid(outputs).view(-1)
            targets_flat = labels.view(-1)
            mask_flat = masks.view(-1).bool()

            # Select valid pixels
            valid_preds = preds_prob[mask_flat]
            valid_targets = targets_flat[mask_flat]

            all_preds.append(valid_preds)
            all_targets.append(valid_targets)

    # Concatenate all batches
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        # Calculate F0.5 score
        val_score = fbeta_score(
            all_preds, all_targets, beta=0.5, threshold=Config.THRESHOLD
        )
    else:
        val_score = 0.0

    return running_loss / len(loader), val_score


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    debug=Config.DEBUG,
    baseline_score=0.551,
):
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        train_df = train_df.head(20)
        val_df = val_df.head(10)
        print(
            f"Debug mode: Training on {len(train_df)} samples, Validating on {len(val_df)} samples."
        )

    # 2. Create Datasets and Loaders
    train_dataset = InkDataset(train_df, mode="train", load_cached_data=True)
    val_dataset = InkDataset(val_df, mode="val", load_cached_data=True)

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
        drop_last=False,
    )

    # 3. Build Model
    device = torch.device(Config.DEVICE)
    model = build_model()
    model.to(device)

    # 4. Optimizer, Scheduler, Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=Config.SCHEDULER_MODE,  # 'max' for F0.5 score
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    criterion = BCETverskyLoss(alpha=0.5, beta=0.5)

    # 5. Training Loop
    best_val_score = -1.0
    early_stopping_counter = 0

    print(f"Starting training for {epochs} epochs...")
    print(f"Baseline F0.5 Score to beat: {baseline_score}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = validate_one_epoch(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_score)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val F0.5: {val_score:.10f}"
        )

        # Validation Gating & Checkpointing
        if val_score > best_val_score:
            best_val_score = val_score
            early_stopping_counter = 0

            # Strict Logic Gate: Only save if we beat the baseline
            if val_score > baseline_score:
                print(
                    f"New best score {val_score:.6f} > baseline {baseline_score}. Saving model..."
                )
                torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                print(
                    f"New best score {val_score:.6f} <= baseline {baseline_score}. Model NOT saved."
                )
        else:
            early_stopping_counter += 1
            print(
                f"Score did not improve. Early stopping counter: {early_stopping_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        # Early Stopping
        if early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation F0.5 Score: {best_val_score:.10f}")
