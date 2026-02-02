import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import seed_everything, fbeta_score, dice_coef
from library.dataset import InkDataset
from library.model import SegFormerB3


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss for semantic segmentation.
    Helps address class imbalance by optimizing for overlap (Dice) and pixel-wise accuracy (BCE).
    """

    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        # preds: Logits from the model (Batch, 1, H, W)
        # targets: Binary ground truth (Batch, 1, H, W)

        # 1. BCE Loss (handles logits internally)
        bce = self.bce_loss(preds, targets)

        # 2. Dice Loss (requires probabilities)
        preds_prob = torch.sigmoid(preds)

        # Flatten tensors
        preds_flat = preds_prob.view(-1)
        targets_flat = targets.view(-1)

        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()

        dice = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Combined Loss
        return self.bce_weight * bce + self.dice_weight * dice


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, F0.5 score, and Dice score.
    """
    model.eval()
    running_loss = 0.0

    # Store predictions and targets for global metric calculation
    # Using lists to collect batches, then concatenating is memory efficient enough for validation
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # Apply sigmoid to convert logits to probabilities for metric calculation
            probs = torch.sigmoid(outputs)

            # Move to CPU to free GPU memory for accumulation
            all_preds.append(probs.cpu())
            all_targets.append(masks.cpu())

    avg_loss = running_loss / len(loader)

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Calculate metrics
    # fbeta_score and dice_coef expect probabilities if threshold is 0.5
    f05 = fbeta_score(all_preds, all_targets, beta=0.5, threshold=0.5)
    dice = dice_coef(all_preds, all_targets, threshold=0.5)

    return avg_loss, f05.item(), dice.item()


def train_model():
    """
    Main training loop with stabilized optimization protocol and validation gating.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: {Config.MODEL_BACKBONE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")

    # 2. Data
    train_dataset = InkDataset(mode="train")
    val_dataset = InkDataset(mode="validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Optimization
    model = SegFormerB3().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Reduce LR if validation metric (F0.5) stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    criterion = BCEDiceLoss()

    # 4. Training Loop
    best_val_score = -1.0
    patience_counter = 0
    model_save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_f05, val_dice = validate(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step(val_f05)

        # Logging (Full precision)
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val F0.5: {val_f05}")
        print(f"Val Dice: {val_dice}")

        # Validation Gating & Checkpointing
        # Only save if we beat the previous best AND the strict baseline
        if val_f05 > best_val_score:
            if val_f05 > Config.BASELINE_SCORE_THRESHOLD:
                print(
                    f"Validation score {val_f05} exceeds baseline {Config.BASELINE_SCORE_THRESHOLD} and previous best {best_val_score}. Saving model."
                )
                torch.save(model.state_dict(), model_save_path)
            else:
                print(
                    f"Validation score {val_f05} improved but is below baseline {Config.BASELINE_SCORE_THRESHOLD}. Not saving."
                )

            best_val_score = val_f05
            patience_counter = 0
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")
