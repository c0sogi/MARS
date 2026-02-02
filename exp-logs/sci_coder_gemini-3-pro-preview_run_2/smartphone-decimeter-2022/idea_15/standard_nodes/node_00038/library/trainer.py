import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import setup_logger


class Trainer:
    """
    Manages the training lifecycle of the SARTransformer model, including
    training loops, validation, early stopping, and model checkpointing.
    """

    def __init__(self, model, train_loader, val_loader, device=None):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (torch.device, optional): Device to run training on.
        """
        self.logger = setup_logger(os.path.join(Config.WORKING_DIR, "trainer.log"))

        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss function: L1 Loss (Mean Absolute Error) is robust to outliers
        self.criterion = nn.L1Loss()

        # Optimizer: AdamW with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4
        )

        # Scheduler: Reduce LR on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.

        Args:
            epoch_idx (int): Current epoch number.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (x_kin, x_sky, y) in enumerate(self.train_loader):
            # Move data to device
            x_kin = x_kin.to(self.device)
            x_sky = x_sky.to(self.device)
            y = y.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(x_kin, x_sky)

            # Compute loss
            loss = self.criterion(preds, y)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate_epoch(self):
        """
        Runs validation on the validation set.

        Returns:
            float: Average validation loss for the epoch.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for x_kin, x_sky, y in self.val_loader:
                # Move data to device
                x_kin = x_kin.to(self.device)
                x_sky = x_sky.to(self.device)
                y = y.to(self.device)

                # Forward pass
                preds = self.model(x_kin, x_sky)

                # Compute loss
                loss = self.criterion(preds, y)

                running_loss += loss.item()

        val_loss = running_loss / len(self.val_loader)
        return val_loss

    def fit(self):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(
            f"Config: Epochs={Config.EPOCHS}, Batch={Config.BATCH_SIZE}, LR={Config.LEARNING_RATE}"
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch()

            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            duration = time.time() - start_time

            # Log metrics with full precision
            self.logger.info(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {duration:.2f}s"
            )

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.info(
                    f"  -> Model saved (Val Loss improved to {best_val_loss:.8f})"
                )
            else:
                patience_counter += 1
                self.logger.info(
                    f"  -> No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            # Early Stopping
            if patience_counter >= Config.PATIENCE:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training complete. Best Val Loss: {best_val_loss:.8f}")

        # Load best model for future use
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )

    def predict(self, test_loader):
        """
        Generates predictions for a given data loader.

        Args:
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            np.ndarray: Predictions array of shape (N, 2).
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                # Handle case where dataset returns (x_kin, x_sky) or (x_kin, x_sky, y)
                if len(batch) == 2:
                    x_kin, x_sky = batch
                else:
                    x_kin, x_sky, _ = batch

                x_kin = x_kin.to(self.device)
                x_sky = x_sky.to(self.device)

                preds = self.model(x_kin, x_sky)
                all_preds.append(preds.cpu().numpy())

        return np.vstack(all_preds)
