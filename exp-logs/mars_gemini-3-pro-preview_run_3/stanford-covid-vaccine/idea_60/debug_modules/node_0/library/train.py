import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import (
    tqdm,
)  # Optional, but good for tracking if allowed, otherwise will minimize print
import time

from library.config import Config
from library.utils import set_seed, scored_mcrmse
from library.data import get_dataloaders
from library.model import HCSDBR_BiGRU
from library.loss import MCRMSELoss


class Trainer:
    """
    Trainer class for the HC-SDBR-BiGRU model.
    Handles training, validation, optimization, and checkpointing.
    """

    def __init__(self, model, train_loader, val_loader, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else Config.DEVICE

        # Move model to device
        self.model.to(self.device)

        # Criterion: MCRMSELoss (Calculates loss on all 5 targets, sliced to 68)
        self.criterion = MCRMSELoss()

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Early Stopping parameters
        self.patience = Config.PATIENCE
        self.best_score = float("inf")
        self.counter = 0

    def train_one_epoch(self, epoch_idx):
        """
        Executes one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        # Iterate over batches
        # Using simple print for progress to avoid clutter if tqdm is not desired,
        # but iterating directly is standard.
        for batch in self.train_loader:
            inputs = batch["inputs"].to(self.device)  # (B, 107, 14)
            targets = batch["targets"].to(self.device)  # (B, 107, 5)
            adj = batch["adjacency"].to(self.device)  # (B, 107)
            mask = batch["mask"].to(self.device)  # (B, 107)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs, adj, mask)  # (B, 107, 5)

            # Compute loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Mandatory for stability)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            # Update weights
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.
        Computes the scored MCRMSE on the specific columns and positions.
        """
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)
                adj = batch["adjacency"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model(inputs, adj, mask)

                # Collect predictions and targets for global metric calculation
                # Move to CPU to save GPU memory
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # Concatenate all batches
        # Shape: (Total_Val_Samples, 107, 5)
        y_pred = torch.cat(all_preds, dim=0)
        y_true = torch.cat(all_targets, dim=0)

        # Calculate Validation Loss (MCRMSE on all 5 targets, sliced to 68)
        # We reuse the criterion logic but on the full dataset
        val_loss = self.criterion(y_pred, y_true).item()

        # Calculate Competition Metric (MCRMSE on 3 scored targets, sliced to 68)
        # scored_mcrmse handles slicing and column filtering internally
        val_metric = scored_mcrmse(y_true, y_pred)

        return val_loss, val_metric

    def fit(self, num_epochs=Config.NUM_EPOCHS):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_metric = self.validate()

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Metric (Scored): {val_metric:.10f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.2f}s"
            )

            # Early Stopping and Model Checkpointing
            # We monitor the competition metric (val_metric)
            if val_metric < self.best_score:
                self.best_score = val_metric
                self.counter = 0
                print(f"New best model found! Saving to {Config.MODEL_SAVE_PATH}")
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

        print(f"Training complete. Best Val Metric: {self.best_score:.10f}")


def run_training(load_cached_data=True, debug=False, epochs=Config.NUM_EPOCHS):
    """
    Orchestrates the training pipeline.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        debug (bool): Whether to run in debug mode (smaller dataset).
        epochs (int): Number of epochs to train.
    """
    # 1. Set Reproducibility
    set_seed(Config.SEED)

    # 2. Load Data
    # The get_dataloaders function handles caching internally based on the flag
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=load_cached_data,
        debug=debug,
    )

    # 3. Initialize Model
    model = HCSDBR_BiGRU()

    # 4. Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader)

    # 5. Execute Training
    trainer.fit(num_epochs=epochs)
