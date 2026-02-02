import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, fbeta_score
from library.losses import BCEDiceLoss
from library.architecture import SegFormerMiTB4
from library.dataset import InkDataset


class Trainer:
    """
    Manages the training and validation lifecycle of the SegFormer model.
    """

    def __init__(self):
        # Ensure reproducibility
        set_seed(Config.SEED)

        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        print(f"Initializing model: {Config.ENCODER_NAME}...")
        self.model = SegFormerMiTB4()
        self.model.to(self.device)

        # Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

        # Loss Function
        self.criterion = BCEDiceLoss()

        # Data Loading
        self._setup_data()

        # State tracking
        self.best_val_score = -float("inf")
        self.current_epoch = 0

    def _setup_data(self):
        """
        Loads metadata and initializes DataLoaders.
        """
        print("Loading metadata...")
        train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)

        # Initialize Datasets
        # We use cached data if available to speed up training
        self.train_dataset = InkDataset(train_df, mode="train", load_cached_data=True)

        self.val_dataset = InkDataset(val_df, mode="validation", load_cached_data=True)

        # Initialize DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

    def train_one_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            # masks = batch['mask'].to(self.device) # Not strictly needed for loss calculation if labels are clean

            self.optimizer.zero_grad()

            # Forward pass (returns logits)
            logits = self.model(images)

            # Calculate loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate(self):
        """
        Runs validation on the validation set.
        Returns average loss and F0.5 score.
        """
        self.model.eval()
        running_loss = 0.0
        running_score = 0.0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                masks = batch["mask"].to(self.device)  # Valid pixel mask

                # Forward pass
                logits = self.model(images)

                # Calculate loss
                loss = self.criterion(logits, labels)
                running_loss += loss.item()

                # Calculate Metric
                # Convert logits to probabilities
                probs = torch.sigmoid(logits)

                # Mask out invalid pixels for metric calculation to be precise
                # (Though labels should be 0 there anyway)
                probs = probs * masks

                # Calculate F0.5 score for this batch
                score = fbeta_score(probs, labels, beta=0.5, threshold=0.5)
                running_score += score

        avg_loss = running_loss / len(self.val_loader)
        avg_score = running_score / len(self.val_loader)

        return avg_loss, avg_score

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training for {Config.EPOCHS} epochs...")
        print(f"Validation Threshold: {Config.VALIDATION_THRESHOLD}")

        start_time = time.time()

        for epoch in range(1, Config.EPOCHS + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch()

            # Validate
            val_loss, val_score = self.validate()

            # Scheduler Step
            self.scheduler.step(val_score)

            epoch_duration = time.time() - epoch_start

            # Print metrics (Full precision as requested)
            print(f"Epoch {epoch}/{Config.EPOCHS} | Time: {epoch_duration:.2f}s")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val F0.5 Score: {val_score}")

            # Validation Gating & Model Saving
            if val_score > self.best_val_score:
                self.best_val_score = val_score

                # Only save if we beat the strict threshold defined in Config
                if val_score > Config.VALIDATION_THRESHOLD:
                    save_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
                    torch.save(self.model.state_dict(), save_path)
                    print(f"New best score! Model saved to {save_path}")
                else:
                    print(
                        f"Score improved but did not exceed threshold {Config.VALIDATION_THRESHOLD}. Model not saved."
                    )

            print("-" * 30)

        total_time = time.time() - start_time
        print(f"Training complete. Total time: {total_time:.2f}s")
        print(f"Best Validation F0.5 Score: {self.best_val_score}")
