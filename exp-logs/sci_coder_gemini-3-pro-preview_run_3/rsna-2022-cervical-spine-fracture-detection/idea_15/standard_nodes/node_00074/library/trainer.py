import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger, seed_everything, get_device
from library.dataset import CervicalSpineDataset, get_transforms
from library.model import ConvNeXtMIL
from library.loss import ImplicitWeightedLoss


class Trainer:
    """
    Orchestrates the training of the Stabilized 2.5D ConvNeXt Multi-Task MIL Network.
    """

    def __init__(self):
        self.logger = get_logger()
        self.device = get_device()
        seed_everything(Config.SEED)

        # --- Data Loading ---
        self.logger.info("Loading metadata...")
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)

        # Debug mode: limit dataset size
        if Config.DEBUG_DATA_SIZE is not None:
            self.logger.info(
                f"Debug mode active: Limiting train/val to {Config.DEBUG_DATA_SIZE} samples."
            )
            train_df = train_df.iloc[: Config.DEBUG_DATA_SIZE]
            val_df = val_df.iloc[: Config.DEBUG_DATA_SIZE]

        self.logger.info(f"Training samples: {len(train_df)}")
        self.logger.info(f"Validation samples: {len(val_df)}")

        # Datasets
        self.train_dataset = CervicalSpineDataset(
            train_df,
            Config.TRAIN_IMAGES_DIR,
            transform=get_transforms(split="train"),
            split="train",
        )
        self.val_dataset = CervicalSpineDataset(
            val_df,
            Config.TRAIN_IMAGES_DIR,
            transform=get_transforms(split="val"),
            split="val",
        )

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,  # Drop last incomplete batch to maintain stability
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # --- Model Setup ---
        self.logger.info(f"Initializing model: {Config.MODEL_NAME}")
        self.model = ConvNeXtMIL(
            model_name=Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            in_channels=Config.IN_CHANNELS,
        )
        self.model.to(self.device)

        # --- Optimization ---
        self.criterion = ImplicitWeightedLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Decoupled Cosine Annealing
        # T_max is set to 1.5x epochs to prevent premature decay
        t_max = int(Config.EPOCHS * Config.T_MAX_MULT)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=t_max, eta_min=1e-6
        )

        # Training State
        self.best_val_loss = float("inf")
        self.patience = 3  # Early stopping patience
        self.counter = 0  # Early stopping counter

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        start_time = time.time()

        for batch_idx, batch_data in enumerate(self.train_loader):
            images = batch_data["image"].to(self.device, dtype=torch.float32)
            targets = batch_data["targets"].to(self.device, dtype=torch.float32)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(images)

            # Loss calculation
            loss = self.criterion(logits, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        elapsed = time.time() - start_time

        current_lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info(
            f"Epoch {epoch_idx+1}/{Config.EPOCHS} | "
            f"Train Loss: {epoch_loss:.8f} | "
            f"LR: {current_lr:.8f} | "
            f"Time: {elapsed:.2f}s"
        )

        return epoch_loss

    def validate(self):
        """
        Runs evaluation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        with torch.no_grad():
            for batch_data in self.val_loader:
                images = batch_data["image"].to(self.device, dtype=torch.float32)
                targets = batch_data["targets"].to(self.device, dtype=torch.float32)
                batch_size = images.size(0)

                logits = self.model(images)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

        val_loss = running_loss / dataset_size
        return val_loss

    def fit(self):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        self.logger.info("Starting training...")

        save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        for epoch in range(Config.EPOCHS):
            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss = self.validate()
            self.logger.info(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Val Loss: {val_loss:.16f}"
            )

            # Step Scheduler
            self.scheduler.step()

            # Checkpointing & Early Stopping
            if val_loss < self.best_val_loss:
                self.logger.info(
                    f"Validation loss improved from {self.best_val_loss:.8f} to {val_loss:.8f}. Saving model..."
                )
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                self.counter = 0
            else:
                self.counter += 1
                self.logger.info(
                    f"Validation loss did not improve. Counter: {self.counter}/{self.patience}"
                )
                if self.counter >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break

        self.logger.info(
            f"Training complete. Best Validation Loss: {self.best_val_loss:.16f}"
        )
