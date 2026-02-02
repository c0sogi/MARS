import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score

from library.config import Config
from library.utils import seed_everything, get_device
from library.loss import AsymmetricLoss
from library.dataset import get_dataloaders
from library.model import ArtworkConvNeXt


class Trainer:
    """
    Trainer class for the Artwork Attribute Labeling task.
    Manages model initialization, training loop, validation, and checkpointing.
    """

    def __init__(self, debug=False):
        """
        Initialize the Trainer with configuration and data.

        Args:
            debug (bool): If True, uses a small subset of data for debugging.
        """
        self.debug = debug
        self.device = get_device()
        self.epochs = Config.NUM_EPOCHS
        self.patience = Config.EARLY_STOPPING_PATIENCE

        # Ensure working directories exist
        Config.setup()

        # Initialize DataLoaders
        self.train_loader, self.val_loader = get_dataloaders(
            debug=self.debug,
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
        )

        # Initialize Model
        self.model = ArtworkConvNeXt(
            model_name=Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
        )
        self.model.to(self.device)

        # Initialize Loss Function (Asymmetric Loss)
        self.criterion = AsymmetricLoss(
            gamma_neg=Config.ASL_GAMMA_NEG,
            gamma_pos=Config.ASL_GAMMA_POS,
            clip=Config.ASL_CLIP,
        )

        # Initialize Optimizer (AdamW)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler (OneCycleLR)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            epochs=self.epochs,
            steps_per_epoch=len(self.train_loader),
            pct_start=0.1,  # 10% warmup
            div_factor=25.0,
            final_div_factor=1000.0,
        )

        # Initialize Mixed Precision Scaler
        self.scaler = GradScaler(enabled=Config.USE_FP16)

        # Early Stopping State
        self.best_val_loss = float("inf")
        self.early_stopping_counter = 0

    def train_one_epoch(self):
        """
        Executes one epoch of training.
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with Mixed Precision
            with autocast(enabled=Config.USE_FP16):
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            # Backward pass with Scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Step Scheduler
            self.scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns:
            tuple: (Average Validation Loss, Micro F1 Score)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                with autocast(enabled=Config.USE_FP16):
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                running_loss += loss.item()

                # Calculate probabilities for metrics
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate Micro F1 using a default threshold of 0.5
        # Note: Advanced threshold optimization is handled in optimize.py
        binary_preds = (all_preds >= 0.5).astype(int)
        val_f1 = f1_score(all_targets, binary_preds, average="micro", zero_division=0)

        return avg_loss, val_f1

    def fit(self):
        """
        Main training loop.
        Iterates through epochs, performs validation, and handles early stopping.
        """
        print(f"Starting training on device: {self.device}")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss, val_f1 = self.validate()

            # Print full precision metrics
            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Micro F1: {val_f1}"
            )

            # Checkpoint and Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"Validation loss improved. Model saved to {Config.MODEL_PATH}")
            else:
                self.early_stopping_counter += 1
                print(
                    f"Validation loss did not improve. "
                    f"Counter: {self.early_stopping_counter}/{self.patience}"
                )

                if self.early_stopping_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        total_time = time.time() - start_time
        print(f"Training finished in {total_time} seconds.")


def run_training(debug=False):
    """
    Helper function to set seeds and run the training pipeline.
    """
    seed_everything(Config.SEED)
    trainer = Trainer(debug=debug)
    trainer.fit()
