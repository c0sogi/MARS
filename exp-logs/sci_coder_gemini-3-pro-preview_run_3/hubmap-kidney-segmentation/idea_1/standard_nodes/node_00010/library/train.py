import os
import torch
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config, seed_everything
from library.dataset import HuBMAPDataset
from library.model import FPNResNet
from library.losses import BCEDiceLoss
from library.utils import dice_coef


def get_transforms(mode="train"):
    """
    Returns the Albumentations transformations for training or validation.
    """
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=1.0
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1.0,
                        ),
                    ],
                    p=0.5,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )


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

        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            # Calculate Dice Score
            # Apply sigmoid to convert logits to probabilities for metric calculation
            preds = torch.sigmoid(outputs)
            # Thresholding is implicit in some dice implementations or we can pass probabilities
            # The library.utils.dice_coef handles continuous probabilities well for soft dice,
            # but for metric reporting we often want a thresholded version or just the soft version.
            # The provided dice_coef doesn't threshold internally, it does (y_true * y_pred).
            # This is Soft Dice.
            batch_dice = dice_coef(masks, preds).item()
            running_dice += batch_dice * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_dice = running_dice / len(loader.dataset)

    return epoch_loss, epoch_dice


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    debug=Config.DEBUG,
):
    """
    Main function to run the training pipeline.
    """
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    if debug:
        print("Debug mode: Using subset of data.")
        train_df = train_df.head(2)
        val_df = val_df.head(2)
        epochs = 2

    # 2. Datasets and Loaders
    train_dataset = HuBMAPDataset(
        metadata_df=train_df,
        mode="train",
        transform=get_transforms("train"),
        load_cached_data=True,
    )

    val_dataset = HuBMAPDataset(
        metadata_df=val_df,
        mode="validation",
        transform=get_transforms("validation"),
        load_cached_data=True,
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

    print(
        f"Training on {len(train_dataset)} tiles, Validating on {len(val_dataset)} tiles."
    )
    print(f"Batch Size: {batch_size}, Tile Size: {Config.TILE_SIZE}")

    # 3. Model, Optimizer, Loss, Scheduler
    device = torch.device(Config.DEVICE)
    model = FPNResNet().to(device)

    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    # Cosine Annealing Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # 4. Training Loop
    best_dice = -1.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{epochs} | LR: {current_lr} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Dice: {val_dice}"
        )

        # Checkpointing
        if val_dice > best_dice:
            print(
                f"Validation Dice improved from {best_dice} to {val_dice}. Saving model..."
            )
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        # Clear cache after validation to prevent fragmentation
        torch.cuda.empty_cache()

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(
                f"Early stopping triggered after {patience_counter} epochs with no improvement."
            )
            break

    print(f"Training complete. Best Validation Dice: {best_dice}")
    print(f"Best model saved to {Config.MODEL_SAVE_PATH}")
