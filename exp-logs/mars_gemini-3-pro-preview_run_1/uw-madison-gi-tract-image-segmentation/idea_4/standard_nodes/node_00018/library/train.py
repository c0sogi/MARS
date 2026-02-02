import os
import time
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import (
    set_seed,
    load_and_preprocess_metadata,
    compute_dice_coefficient,
)
from library.dataset import UWDataset, get_transforms
from library.model import UnetPlusPlus
from library.loss import DeepSupervisionLoss


class Trainer:
    """
    Manages the training and validation lifecycle of the U-Net++ model.
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.working_dir = Config.WORKING_DIR
        self.checkpoint_dir = Config.CHECKPOINT_DIR

        # Ensure directories exist
        os.makedirs(self.working_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize Model
        print(f"Initializing {Config.ARCH} with {Config.BACKBONE} backbone...")
        self.model = UnetPlusPlus(
            backbone_name=Config.BACKBONE,
            classes=Config.NUM_CLASSES,
            deep_supervision=Config.DEEP_SUPERVISION,
        )
        self.model.to(self.device)

        # Optimization Components
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # Loss Function
        self.criterion = DeepSupervisionLoss(
            bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler()

    def train_one_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["image"].to(self.device, dtype=torch.float32)
            masks = batch["mask"].to(self.device, dtype=torch.float32)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast():
                # In training mode with deep supervision, output is a list of tensors
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # Backward Pass and Optimizer Step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs validation on the validation set.
        Returns the mean Dice coefficient across all classes and samples.
        """
        self.model.eval()
        dice_scores = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device, dtype=torch.float32)
                masks = batch["mask"].to(self.device, dtype=torch.float32)

                # In eval mode, model returns a single tensor (final output)
                outputs = self.model(images)

                # Apply Sigmoid to logits to get probabilities
                preds = torch.sigmoid(outputs)

                # Threshold predictions for Dice calculation
                preds_binary = (preds > 0.5).float()

                # Compute Dice for this batch
                # compute_dice_coefficient expects flattened arrays or tensors
                # We calculate it per batch to average later
                batch_dice = compute_dice_coefficient(masks, preds_binary)
                dice_scores.append(batch_dice)

        mean_dice = np.mean(dice_scores)
        return mean_dice

    def fit(self, debug=Config.DEBUG, epochs=Config.EPOCHS):
        """
        Main training loop.

        Args:
            debug (bool): If True, runs on a small subset of data.
            epochs (int): Number of training epochs.
        """
        set_seed(Config.SEED)

        # 1. Load Data
        print("Loading metadata...")
        df_train = load_and_preprocess_metadata(Config.TRAIN_CSV)
        df_val = load_and_preprocess_metadata(Config.VAL_CSV)

        if debug:
            print("Debug mode: Subsampling data...")
            df_train = df_train.head(Config.BATCH_SIZE * 2)
            df_val = df_val.head(Config.BATCH_SIZE * 2)

        # 2. Create Datasets and Loaders
        train_dataset = UWDataset(
            df_train, mode="train", transforms=get_transforms("train")
        )
        val_dataset = UWDataset(df_val, mode="val", transforms=get_transforms("val"))

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
            drop_last=False,
        )

        print(f"Starting training for {epochs} epochs...")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        best_dice = 0.0
        patience_counter = 0
        best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train Step
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validation Step
            val_dice = self.validate(val_loader)

            # Scheduler Step
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Logging
            print(
                f"Epoch {epoch}/{epochs} - "
                f"Time: {elapsed:.2f}s - "
                f"LR: {current_lr:.2e} - "
                f"Train Loss: {train_loss:.16f} - "
                f"Val Dice: {val_dice:.16f}"
            )

            # Checkpointing
            if val_dice > best_dice:
                print(
                    f"Validation Dice improved from {best_dice:.16f} to {val_dice:.16f}. Saving model..."
                )
                best_dice = val_dice
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_dice": best_dice,
                    },
                    best_model_path,
                )
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            # Early Stopping
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered. Training finished.")
                break

        print(f"Training complete. Best Validation Dice: {best_dice:.16f}")
        return best_dice
