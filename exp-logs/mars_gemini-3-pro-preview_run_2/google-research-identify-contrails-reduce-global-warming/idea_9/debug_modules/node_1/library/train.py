import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloader
from library.model import HybridResNetTransformerUNet
from library.loss import HybridLoss


class Trainer:
    """
    Manages the training and validation lifecycle of the Hybrid ResNet18-Transformer U-Net.
    """

    def __init__(self, train_loader, val_loader, device=Config.DEVICE):
        """
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (torch.device): Device to run training on.
        """
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize Model
        self.model = HybridResNetTransformerUNet()
        self.model.to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Loss Function
        self.criterion = HybridLoss(
            bce_weight=Config.BCE_WEIGHT, dice_weight=Config.DICE_WEIGHT
        )

        # Training State
        self.best_score = float("-inf")
        self.early_stopping_patience = 10
        self.early_stopping_counter = 0

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device, dtype=torch.float)
            masks = masks.to(self.device, dtype=torch.float)
            batch_size = images.size(0)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(images)

            # Calculate loss
            loss = self.criterion(logits, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self):
        """
        Runs validation and calculates Global Dice Score.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        # Accumulators for Global Dice
        # Dice = 2 * |X n Y| / (|X| + |Y|)
        total_intersection = 0.0
        total_union = 0.0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, dtype=torch.float)
                masks = masks.to(self.device, dtype=torch.float)
                batch_size = images.size(0)

                # Forward pass
                logits = self.model(images)

                # Calculate Validation Loss
                loss = self.criterion(logits, masks)
                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Calculate Global Dice Components
                # Apply sigmoid and threshold
                preds = (torch.sigmoid(logits) > Config.THRESHOLD).float()

                # Flatten for calculation
                preds_flat = preds.view(-1)
                targets_flat = masks.view(-1)

                intersection = (preds_flat * targets_flat).sum().item()
                union = preds_flat.sum().item() + targets_flat.sum().item()

                total_intersection += intersection
                total_union += union

        val_loss = running_loss / dataset_size

        # Compute Global Dice
        # Add small epsilon to avoid division by zero if empty
        global_dice = (2.0 * total_intersection) / (total_union + 1e-6)

        return val_loss, global_dice

    def fit(self, epochs=Config.EPOCHS):
        """
        Main training loop with Early Stopping and Model Checkpointing.
        """
        print(f"Starting training for {epochs} epochs on {self.device}...")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_dice = self.validate()

            # Scheduler Step
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            epoch_duration = time.time() - epoch_start

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{epochs} | Time: {epoch_duration:.2f}s | LR: {current_lr:.2e}"
            )
            print(f"  Train Loss: {train_loss}")
            print(f"  Val Loss:   {val_loss}")
            print(f"  Val Dice:   {val_dice}")

            # Model Checkpointing
            if val_dice > self.best_score:
                print(
                    f"  Score Improved ({self.best_score} -> {val_dice}). Saving model to {Config.MODEL_PATH}..."
                )
                self.best_score = val_dice
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                print(
                    f"  Score did not improve. Early stopping counter: {self.early_stopping_counter}/{self.early_stopping_patience}"
                )

            # Early Stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print("Early stopping triggered. Training stopped.")
                break

        total_time = time.time() - start_time
        print(
            f"Training complete in {total_time:.2f}s. Best Validation Dice: {self.best_score}"
        )


def train_model(
    max_train_samples=Config.MAX_TRAIN_SAMPLES,
    max_val_samples=Config.MAX_VAL_SAMPLES,
    epochs=Config.EPOCHS,
):
    """
    Initializes data loaders and starts the training process.

    Args:
        max_train_samples (int, optional): Limit training data size for debugging.
        max_val_samples (int, optional): Limit validation data size for debugging.
        epochs (int): Number of epochs to train.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Prepare DataLoaders
    print("Initializing DataLoaders...")
    train_loader = get_dataloader(
        split="train", batch_size=Config.BATCH_SIZE, max_samples=max_train_samples
    )

    val_loader = get_dataloader(
        split="validation", batch_size=Config.BATCH_SIZE, max_samples=max_val_samples
    )

    # Initialize Trainer
    trainer = Trainer(train_loader, val_loader)

    # Start Training
    trainer.fit(epochs=epochs)
