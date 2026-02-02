import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import get_device, compute_mae, seed_everything
from library.model import NCPNet


class Trainer:
    """
    Manages the training, validation, and checkpointing of the NCP-Net model.
    """

    def __init__(self, model=None):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module, optional): A pre-instantiated model. If None, initializes NCPNet.
        """
        self.device = get_device()
        self.model = model if model is not None else NCPNet()
        self.model.to(self.device)

        # Optimization components
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )

        # Best score tracking
        self.best_val_mae = float("inf")
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training.

        Args:
            train_loader (DataLoader): The training data loader.
            epoch_idx (int): Current epoch index (for logging).

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            # Move data to device
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            u_out = batch["u_out"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Model output shape: (Batch, Seq, 1)
            preds = self.model(x)

            # --- Masked L1 Loss Calculation ---
            # We only care about error when u_out == 0 (Inspiratory phase)
            # u_out is (Batch, Seq), preds/y are (Batch, Seq, 1)

            # Ensure mask matches prediction shape
            mask = (u_out == 0).float().unsqueeze(-1)

            # Calculate absolute error
            abs_error = torch.abs(preds - y)

            # Apply mask
            masked_error = abs_error * mask

            # Compute mean only over valid elements
            # Add epsilon to denominator to prevent division by zero
            loss = masked_error.sum() / (mask.sum() + 1e-8)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.CLIP_GRAD_NORM
            )

            # Optimizer step
            self.optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
        return avg_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.

        Args:
            val_loader (DataLoader): The validation data loader.

        Returns:
            float: The Mean Absolute Error (MAE) on the inspiratory phase.
        """
        self.model.eval()
        total_mae = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                u_out = batch["u_out"].to(self.device)

                preds = self.model(x)

                # Use the utility function for consistent metric calculation
                # compute_mae expects tensors or arrays
                batch_mae = compute_mae(preds, y, u_out)

                total_mae += batch_mae
                total_batches += 1

        avg_mae = total_mae / total_batches if total_batches > 0 else 0.0
        return avg_mae

    def fit(
        self, train_loader, val_loader, epochs=Config.EPOCHS, early_stopping_patience=15
    ):
        """
        Main training loop with Early Stopping and Scheduler.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
            early_stopping_patience (int): Epochs to wait without improvement before stopping.
        """
        print(f"Starting training on device: {self.device}")
        print(f"Epochs: {epochs}, Batch Size: {Config.BATCH_SIZE}")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_mae = self.validate(val_loader)

            # Scheduler step
            self.scheduler.step(val_mae)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val MAE: {val_mae:.8f} | "
                f"LR: {current_lr:.2e}"
            )

            # Checkpoint and Early Stopping Logic
            if val_mae < self.best_val_mae:
                print(
                    f"Validation MAE improved ({self.best_val_mae:.8f} -> {val_mae:.8f}). Saving model..."
                )
                self.best_val_mae = val_mae
                self.save_model(self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{early_stopping_patience}"
                )

            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation MAE: {self.best_val_mae:.8f}")

    def save_model(self, path):
        """
        Saves the model state dictionary.
        """
        torch.save(self.model.state_dict(), path)

    def load_best_model(self):
        """
        Loads the best model state dictionary from disk.
        """
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print(f"Loaded best model from {self.best_model_path}")
        else:
            print(f"Warning: No model found at {self.best_model_path}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set.

        Args:
            test_loader (DataLoader): Test data loader.

        Returns:
            np.ndarray: Flat array of predictions.
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(self.device)
                # Forward pass
                preds = self.model(x)

                # Flatten predictions (Batch, Seq, 1) -> (Batch * Seq)
                preds_flat = preds.view(-1).cpu().numpy()
                predictions.append(preds_flat)

        return np.concatenate(predictions)
