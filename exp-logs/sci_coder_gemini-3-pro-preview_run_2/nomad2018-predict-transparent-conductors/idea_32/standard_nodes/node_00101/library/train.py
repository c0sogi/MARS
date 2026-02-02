import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.model import MS_RA_CGN
from library.data import get_dataloaders
from library.utils import set_seed, TargetScaler, rmsle


class Trainer:
    """
    Manages the training, validation, and checkpointing of the MS-RA-CGN model.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.model = MS_RA_CGN().to(self.device)

        # Optimizer with decoupled weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

        # Loss function (MSE on standardized targets)
        self.criterion = nn.MSELoss()

        # Target scaler for standardization
        self.scaler = TargetScaler()

    def fit_scaler(self, train_loader):
        """
        Fits the TargetScaler on the entire training dataset and saves the state.
        """
        print("Fitting TargetScaler...")
        all_y = []
        for batch in train_loader:
            all_y.append(batch.y)

        all_y = torch.cat(all_y, dim=0)
        self.scaler.fit(all_y)

        # Save scaler state for inference
        scaler_path = os.path.join(Config.CACHE_DIR, "target_scaler.npz")
        self.scaler.save(scaler_path)

    def train_one_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)

            # Standardize targets
            y_target = self.scaler.transform(batch.y)

            self.optimizer.zero_grad()
            y_pred = self.model(batch)

            # Calculate loss
            loss = self.criterion(y_pred, y_target)

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average MSE loss (scaled) and RMSLE (original scale).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)

                # Forward pass
                y_pred_scaled = self.model(batch)

                # Calculate MSE on scaled targets for scheduler/loss tracking
                y_target_scaled = self.scaler.transform(batch.y)
                loss = self.criterion(y_pred_scaled, y_target_scaled)
                total_loss += loss.item()

                # Inverse transform for RMSLE calculation (back to eV)
                y_pred = self.scaler.inverse_transform(y_pred_scaled)

                all_preds.append(y_pred.cpu())
                all_targets.append(batch.y.cpu())
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Calculate RMSLE on original scale
        y_pred_all = torch.cat(all_preds, dim=0)
        y_true_all = torch.cat(all_targets, dim=0)
        val_rmsle = rmsle(y_true_all, y_pred_all)

        return avg_loss, val_rmsle

    def run(self, load_cached_data=True):
        """
        Main training loop.
        """
        set_seed(Config.SEED)

        # Get dataloaders
        train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

        # Fit scaler on training data
        self.fit_scaler(train_loader)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

        print(f"\nStarting training on {Config.DEVICE}...")
        print(f"Max Epochs: {Config.MAX_EPOCHS}, Patience: {Config.PATIENCE}")

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            start_time = time.time()

            # Training step
            train_loss = self.train_one_epoch(train_loader)

            # Validation step
            val_loss, val_rmsle = self.validate(val_loader)

            # Scheduler step (monitor validation MSE)
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch:03d}/{Config.MAX_EPOCHS} "
                f"| Train MSE: {train_loss:.6f} "
                f"| Val MSE: {val_loss:.6f} "
                f"| Val RMSLE: {val_rmsle:.6f} "
                f"| LR: {current_lr:.2e} "
                f"| Time: {epoch_time:.2f}s"
            )

            # Checkpointing and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  -> New best model saved to {best_model_path}")
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

        print(f"\nTraining complete. Best Validation MSE: {best_val_loss:.6f}")


def train_model(load_cached_data=True):
    """
    Wrapper function to initialize Trainer and start training.
    """
    trainer = Trainer()
    trainer.run(load_cached_data=load_cached_data)
