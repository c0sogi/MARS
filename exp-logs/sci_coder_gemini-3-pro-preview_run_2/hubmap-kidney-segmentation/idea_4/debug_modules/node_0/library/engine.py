import os
import time
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from library files
from library.config import Config
from library.utils import seed_everything, get_score
from library.loss import MultiTaskLoss
from library.model import MultiTaskResNetFPN
from library.data import prepare_tiles, HuBMAPDataset, get_transforms


class Trainer:
    def __init__(self):
        self.config = Config
        seed_everything(self.config.SEED)

        self.device = torch.device(self.config.DEVICE)

        # Initialize Model
        self.model = MultiTaskResNetFPN()
        self.model.to(self.device)

        # Initialize Loss
        self.criterion = MultiTaskLoss()

        # Initialize Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Scheduler will be initialized in fit() after knowing the number of epochs
        self.scheduler = None

        # Best score tracking
        self.best_score = -np.inf

    def train_fn(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for step, (images, masks) in enumerate(train_loader):
            images = images.to(self.device, dtype=torch.float)
            masks = masks.to(self.device, dtype=torch.float)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def valid_fn(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        preds = []
        targets = []

        with torch.no_grad():
            for step, (images, masks) in enumerate(val_loader):
                images = images.to(self.device, dtype=torch.float)
                masks = masks.to(self.device, dtype=torch.float)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                running_loss += loss.item()

                # For metric calculation, we only care about the Primary Head (Channel 0)
                # Output shape: (B, 2, H, W) -> Select channel 0 -> (B, H, W)
                # Apply sigmoid to logits for scoring
                primary_logits = outputs[:, 0, :, :]
                primary_probs = torch.sigmoid(primary_logits)

                # Target shape: (B, 2, H, W) -> Select channel 0 -> (B, H, W)
                primary_targets = masks[:, 0, :, :]

                # Store for global metric calculation (on CPU to save GPU memory)
                preds.append(primary_probs.cpu())
                targets.append(primary_targets.cpu())

        # Concatenate all batches
        preds = torch.cat(preds)
        targets = torch.cat(targets)

        # Calculate Dice Score
        # We use a threshold of 0.5 (implicit in get_score logic if inputs are probabilities)
        # get_score expects logits or probs. If probs, it thresholds.
        # Since we passed probs, get_score will handle it.
        val_dice = get_score(preds, targets, threshold=self.config.MASK_THRESHOLD)

        return running_loss / len(val_loader), val_dice

    def fit(self, load_cached_data=True):
        print(f"Starting training with device: {self.device}")

        # 1. Load Metadata
        train_df = pd.read_csv(
            os.path.join(self.config.METADATA_DIR, "train_metadata.csv")
        )
        val_df = pd.read_csv(os.path.join(self.config.METADATA_DIR, "val_metadata.csv"))

        if self.config.DEBUG:
            train_df = train_df.head(self.config.DEBUG_SAMPLE_SIZE)
            val_df = val_df.head(self.config.DEBUG_SAMPLE_SIZE)
            print("Debug mode: Reduced dataset size.")

        # 2. Prepare Tiles
        train_tiles = prepare_tiles(
            train_df, mode="train", load_cached_data=load_cached_data
        )
        val_tiles = prepare_tiles(val_df, mode="val", load_cached_data=load_cached_data)

        # 3. Datasets and Loaders
        train_dataset = HuBMAPDataset(
            train_tiles, transforms=get_transforms(mode="train"), mode="train"
        )
        val_dataset = HuBMAPDataset(
            val_tiles, transforms=get_transforms(mode="val"), mode="val"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # 4. Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.config.T_MAX, eta_min=self.config.MIN_LR
        )

        # 5. Training Loop
        early_stopping_counter = 0

        for epoch in range(self.config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_fn(train_loader)

            # Validate
            val_loss, val_dice = self.valid_fn(val_loader)

            # Step Scheduler
            self.scheduler.step()

            elapsed = time.time() - start_time

            print(f"Epoch {epoch+1}/{self.config.EPOCHS} - Time: {elapsed:.0f}s")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Dice: {val_dice}")

            # Warmup Logic
            if (epoch + 1) <= self.config.WARMUP_EPOCHS:
                print(
                    f"Warmup Phase ({epoch+1}/{self.config.WARMUP_EPOCHS}): Best model saving and early stopping disabled."
                )
                continue

            # Checkpointing
            if val_dice > self.best_score:
                print(
                    f"Validation Dice improved ({self.best_score} -> {val_dice}). Saving model..."
                )
                self.best_score = val_dice
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(
                    f"No improvement. Early stopping counter: {early_stopping_counter}/{self.config.PATIENCE}"
                )

            # Early Stopping
            if early_stopping_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val Dice: {self.best_score}")
