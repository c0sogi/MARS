import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.model import VentilatorModel
from library.loss import MaskedL1Loss


def set_seed(seed: int):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Trainer class for the Ventilator Pressure Prediction task.
    Handles training, validation, optimization, and checkpointing.
    """

    def __init__(self, config: Config, model: VentilatorModel):
        """
        Args:
            config (Config): Configuration object.
            model (VentilatorModel): The model to train.
        """
        self.config = config
        self.device = torch.device(self.config.DEVICE)
        self.model = model.to(self.device)
        self.criterion = MaskedL1Loss(aux_weight=self.config.AUX_WEIGHT)

        # Ensure reproducibility
        set_seed(self.config.SEED)

    def train_epoch(
        self, train_loader: DataLoader, optimizer: optim.Optimizer, scheduler
    ) -> float:
        """
        Runs one epoch of training.

        Args:
            train_loader (DataLoader): Training data.
            optimizer (Optimizer): PyTorch optimizer.
            scheduler: Learning rate scheduler.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch in train_loader:
            # Move data to device
            inputs = batch["input"].to(self.device)
            u_out = batch["u_out"].to(self.device)
            targets = batch["target"].to(self.device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # Model returns (final_pred, aux_pred)
            final_pred, aux_pred = self.model(inputs, u_out=u_out)

            # Calculate Loss
            loss = self.criterion(final_pred, aux_pred, targets, u_out)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Critical for Wide-State LSTM stability)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.CLIP_GRAD
            )

            # Optimizer Step
            optimizer.step()

            # Scheduler Step (OneCycleLR steps per batch)
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()

        return total_loss / num_batches

    def validate(self, val_loader: DataLoader) -> float:
        """
        Runs validation and calculates the competition metric (MAE on inspiratory phase).

        Args:
            val_loader (DataLoader): Validation data.

        Returns:
            float: Mean Absolute Error on the inspiratory phase.
        """
        self.model.eval()
        total_mae = 0.0
        total_count = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(self.device)
                u_out = batch["u_out"].to(self.device)
                targets = batch["target"].to(self.device)

                # Forward pass (only need final prediction for metric)
                final_pred, _ = self.model(inputs, u_out=u_out)

                # Ensure shapes match (Batch, Seq, 1)
                if targets.dim() == 2:
                    targets = targets.unsqueeze(-1)
                if u_out.dim() == 2:
                    u_out = u_out.unsqueeze(-1)

                # Calculate Absolute Error
                abs_error = torch.abs(final_pred - targets)

                # Mask: Only score inspiratory phase (u_out == 0)
                mask = 1.0 - u_out
                masked_error = abs_error * mask

                # Accumulate
                total_mae += masked_error.sum().item()
                total_count += mask.sum().item()

        # Avoid division by zero
        if total_count == 0:
            return 0.0

        return total_mae / total_count

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, patience: int = 7):
        """
        Main training loop with Early Stopping and Scheduler.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            patience (int): Number of epochs to wait for improvement before stopping.
        """
        print(f"Starting training on device: {self.device}")

        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LR_MAX,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Scheduler (OneCycleLR)
        # Steps per epoch is required for OneCycleLR
        steps_per_epoch = len(train_loader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.LR_MAX,
            epochs=self.config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.config.PCT_START,
            anneal_strategy="cos",
        )

        # Early Stopping State
        best_val_mae = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(self.config.WORKING_DIR, "model.pth")

        for epoch in range(1, self.config.EPOCHS + 1):
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)

            # Validate
            val_mae = self.validate(val_loader)

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss} | "
                f"Val MAE: {val_mae}"
            )

            # Early Stopping Check
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Val MAE: {best_val_mae}")

        # Load best model for future use (e.g. inference)
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )
