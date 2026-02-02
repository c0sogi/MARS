import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, metric_mcrmse
from library.dataset import get_data, RNADataset
from library.model import RNA_Net
from library.loss import DeepSupervisionLoss


class Trainer:
    """
    Trainer class for the RNA Degradation Prediction task.
    Handles model training, validation, and checkpointing.
    """

    def __init__(self, device=None):
        """
        Initialize the Trainer.

        Args:
            device (torch.device, optional): Compute device. Defaults to auto-detect.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize Model
        self.model = RNA_Net().to(self.device)

        # Initialize Loss
        self.criterion = DeepSupervisionLoss()

        # Initialize Optimizer
        # Using AdamW with low weight decay as specified in Idea
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        # Cosine Annealing over the fixed number of epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS
        )

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.

        Args:
            dataloader (DataLoader): Training data loader.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for seq, loop, dist, targets in dataloader:
            # Move data to device
            seq = seq.to(self.device)
            loop = loop.to(self.device)
            dist = dist.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Returns main_pred and list of layer_preds
            main_pred, layer_preds = self.model(seq, loop, dist)

            # Compute Loss (Deep Supervision)
            loss = self.criterion(main_pred, layer_preds, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Critical for BiLSTM stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            # Optimizer step
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self, dataloader):
        """
        Runs validation and calculates MCRMSE.

        Args:
            dataloader (DataLoader): Validation data loader.

        Returns:
            float: MCRMSE score.
        """
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for seq, loop, dist, targets in dataloader:
                seq = seq.to(self.device)
                loop = loop.to(self.device)
                dist = dist.to(self.device)

                # Forward pass
                # We only care about the main prediction for evaluation
                main_pred, _ = self.model(seq, loop, dist)

                # Move to CPU
                main_pred = main_pred.cpu().numpy()
                targets = targets.cpu().numpy()

                # Slice predictions to scored length (68)
                # Targets are already sliced/padded in dataset, but let's be safe
                # Config.PRED_LEN is 68
                pred_len = Config.PRED_LEN
                main_pred = main_pred[:, :pred_len, :]
                targets = targets[:, :pred_len, :]

                all_preds.append(main_pred)
                all_targets.append(targets)

        # Concatenate all batches
        if not all_preds:
            return 0.0

        y_pred = np.concatenate(all_preds, axis=0)
        y_true = np.concatenate(all_targets, axis=0)

        # Calculate MCRMSE
        score = metric_mcrmse(y_true, y_pred)
        return score

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS, patience=5):
        """
        Main training loop with Early Stopping and Checkpointing.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
        """
        print(f"Starting training on device: {self.device}")
        best_score = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_score = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Print metrics (Full precision as requested)
            print(
                f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
            )

            # Checkpointing
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                print(f"New best model found! Saving to {Config.MODEL_PATH}")
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete. Best Validation MCRMSE: {best_score}")


def train(load_cached_data=True):
    """
    Orchestrates the training process.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Prepare Data
    # get_data handles caching logic internally
    train_data = get_data(mode="train", load_cached_data=load_cached_data)
    val_data = get_data(mode="val", load_cached_data=load_cached_data)

    train_dataset = RNADataset(train_data)
    val_dataset = RNADataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Initialize Trainer
    trainer = Trainer()

    # 4. Start Training
    # Using a patience of 5 for early stopping
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS, patience=5)
