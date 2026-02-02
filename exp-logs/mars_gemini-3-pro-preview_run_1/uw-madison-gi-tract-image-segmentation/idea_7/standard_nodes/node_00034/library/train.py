import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    CHECKPOINT_DIR,
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    BCE_WEIGHT,
    DICE_WEIGHT,
    NUM_CLASSES,
    SEED,
)
from library.config import set_seed
from library.dataset import get_dataloaders
from library.model import RecurrentUNet


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        # BCE Loss
        bce_loss = self.bce(pred, target)

        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        smooth = 1e-5

        # Flatten for calculation: (B, C, H, W) -> (B, C, H*W)
        # We calculate Dice per channel per sample, then average
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class Trainer:
    def __init__(
        self, model, loaders, criterion, optimizer, scheduler, device, patience=5
    ):
        self.model = model
        self.loaders = loaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.best_score = -float("inf")
        self.counter = 0
        self.best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        start_time = time.time()

        for batch_idx, (images, masks) in enumerate(self.loaders["train"]):
            images = images.to(self.device)
            masks = masks.to(self.device)

            batch_size = images.size(0)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1000)

            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        elapsed = time.time() - start_time

        print(f"Epoch {epoch+1} Train Loss: {epoch_loss:.6f} Time: {elapsed:.0f}s")
        return epoch_loss

    def validate(self, epoch):
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0
        dataset_size = 0

        with torch.no_grad():
            for images, masks in self.loaders["val"]:
                images = images.to(self.device)
                masks = masks.to(self.device)

                batch_size = images.size(0)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # Calculate metric (Dice) for monitoring
                pred_sigmoid = torch.sigmoid(outputs)
                pred_binary = (pred_sigmoid > 0.5).float()

                smooth = 1e-5
                intersection = (pred_binary * masks).sum(dim=(2, 3))
                union = pred_binary.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                dice = (2.0 * intersection + smooth) / (union + smooth)

                running_loss += loss.item() * batch_size
                running_dice += dice.mean().item() * batch_size
                dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_dice = running_dice / dataset_size

        print(f"Epoch {epoch+1} Val Loss: {epoch_loss:.6f} Val Dice: {epoch_dice:.10f}")
        return epoch_loss, epoch_dice

    def fit(self, epochs):
        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            self.train_epoch(epoch)
            val_loss, val_dice = self.validate(epoch)

            if self.scheduler:
                self.scheduler.step()

            # Checkpointing and Early Stopping
            if val_dice > self.best_score:
                print(
                    f"Validation Dice improved from {self.best_score:.6f} to {val_dice:.6f}. Saving model..."
                )
                self.best_score = val_dice
                torch.save(self.model.state_dict(), self.best_model_path)
                self.counter = 0
            else:
                self.counter += 1
                print(
                    f"No improvement. EarlyStopping counter: {self.counter}/{self.patience}"
                )
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Dice: {self.best_score:.6f}")


def train_model(debug=False, epochs=EPOCHS):
    set_seed(SEED)

    # 1. Load Metadata
    print("Loading metadata...")
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        raise FileNotFoundError(
            "Metadata CSV files not found. Please ensure metadata generation is complete."
        )

    df_train = pd.read_csv(TRAIN_CSV, keep_default_na=False)
    df_val = pd.read_csv(VAL_CSV, keep_default_na=False)

    if debug:
        print("Debug mode enabled. Using subset of data.")
        # Sample by case to maintain integrity if possible, or just head
        df_train = df_train.head(200)
        df_val = df_val.head(50)

    print(f"Train samples: {len(df_train)}, Val samples: {len(df_val)}")

    # 2. Prepare DataLoaders
    loaders = get_dataloaders(df_train, df_val)

    # 3. Initialize Model
    print("Initializing Recurrent U-Net...")
    model = RecurrentUNet(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(DEVICE)

    # 4. Setup Training Components
    criterion = BCEDiceLoss(bce_weight=BCE_WEIGHT, dice_weight=DICE_WEIGHT)

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 5. Start Training
    trainer = Trainer(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        patience=5,
    )

    trainer.fit(epochs)
