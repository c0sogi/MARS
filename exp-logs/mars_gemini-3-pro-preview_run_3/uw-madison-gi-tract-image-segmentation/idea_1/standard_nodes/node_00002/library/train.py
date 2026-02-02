import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import set_seed, dice_coef
from library.dataset import (
    process_and_cache_25d_metadata,
    GI_MRI_Dataset,
    get_transforms,
)
from library.model import FPN


class CombinedLoss(nn.Module):
    """
    Combined Loss: Weighted sum of BCE Loss and Dice Loss.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCELoss()

    def forward(self, inputs, targets):
        # Clamp inputs to prevent log(0) in BCELoss since inputs are already sigmoid probabilities
        inputs_clamped = torch.clamp(inputs, min=1e-7, max=1 - 1e-7)

        bce = self.bce_loss(inputs_clamped, targets)

        # Calculate Dice Loss (1 - Dice Coefficient)
        # dice_coef returns a value between 0 and 1
        dice = dice_coef(inputs, targets)
        dice_loss = 1.0 - dice

        return self.bce_weight * bce + self.dice_weight * dice_loss


class Trainer:
    """
    Trainer class to manage the training and validation loops.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        patience=5,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.best_dice = 0.0
        self.best_epoch = 0
        self.save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_dice = 0.0
        dataset_size = 0

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            # Metrics
            dice = dice_coef(outputs.detach(), masks.detach())

            running_loss += loss.item() * batch_size
            running_dice += dice.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_dice = running_dice / dataset_size

        return epoch_loss, epoch_dice

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        running_dice = 0.0
        dataset_size = 0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                batch_size = images.size(0)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice = dice_coef(outputs, masks)

                running_loss += loss.item() * batch_size
                running_dice += dice.item() * batch_size
                dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_dice = running_dice / dataset_size

        return epoch_loss, epoch_dice

    def fit(self, epochs):
        print(f"Starting training on device: {self.device}")
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss, train_dice = self.train_one_epoch(epoch)
            val_loss, val_dice = self.validate()

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(
                f"Epoch {epoch}/{epochs} | Time: {int(epoch_mins)}m {int(epoch_secs)}s"
            )
            print(f"  Train Loss: {train_loss:.8f} | Train Dice: {train_dice:.8f}")
            print(f"  Val Loss:   {val_loss:.8f} | Val Dice:   {val_dice:.8f}")

            # Checkpoint and Early Stopping
            if val_dice > self.best_dice:
                print(
                    f"  [Improvement] Val Dice increased from {self.best_dice:.8f} to {val_dice:.8f}. Saving model..."
                )
                self.best_dice = val_dice
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), self.save_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(
                    f"  [No Improvement] Counter: {epochs_no_improve}/{self.patience}"
                )

            if epochs_no_improve >= self.patience:
                print(
                    f"Early stopping triggered. Best Val Dice: {self.best_dice:.8f} at Epoch {self.best_epoch}"
                )
                break

        print(f"Training complete. Best model saved to {self.save_path}")
        return self.best_dice


def train_model(debug=None):
    """
    Main function to setup and run the training process.
    """
    # 1. Setup Configuration
    Config.setup(debug=debug, training=True)
    set_seed(Config.SEED)

    # 2. Load Metadata
    print("Loading metadata...")
    train_df, val_df, _ = process_and_cache_25d_metadata(load_cached_data=True)

    if Config.DEBUG:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 3. Create Datasets and Dataloaders
    print("Creating datasets...")
    train_dataset = GI_MRI_Dataset(
        df=train_df, transforms=get_transforms(mode="train"), mode="train"
    )
    val_dataset = GI_MRI_Dataset(
        df=val_df, transforms=get_transforms(mode="val"), mode="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 4. Initialize Model, Loss, Optimizer
    print("Initializing model...")
    device = torch.device(Config.DEVICE)
    model = FPN(
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    )
    model = model.to(device)

    criterion = CombinedLoss(
        bce_weight=Config.LOSS_WEIGHT_BCE, dice_weight=Config.LOSS_WEIGHT_DICE
    )

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 5. Start Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        patience=5,  # Stop if no improvement for 5 epochs
    )

    best_score = trainer.fit(epochs=Config.EPOCHS)
    return best_score
