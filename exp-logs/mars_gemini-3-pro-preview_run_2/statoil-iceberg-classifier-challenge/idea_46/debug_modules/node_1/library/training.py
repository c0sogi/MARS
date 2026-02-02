import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import get_logger

# Initialize module-level logger
logger = get_logger("Trainer")


class Trainer:
    """
    Manages the training lifecycle for the Triple-Stream Wide-Body Network.
    Handles optimization, validation, scheduling, and early stopping.
    """

    def __init__(self, model, train_loader, val_loader, device=None):
        """
        Args:
            model (nn.Module): The TS-WBN model.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            device (str, optional): Computation device ('cuda' or 'cpu').
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else Config.DEVICE

        # Move model to device
        self.model.to(self.device)

        # Optimization Components
        # Strategy: "Low and Slow" with Adam (not AdamW)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

        # Criterion: Binary Cross Entropy with Logits
        # The model outputs raw logits (no sigmoid), so we use BCEWithLogitsLoss
        self.criterion = nn.BCEWithLogitsLoss()

        # State Management
        self.best_model_state = None
        self.best_val_loss = float("inf")

    def train_one_epoch(self, epoch_index):
        """
        Executes one training epoch.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (images, angles, labels) in enumerate(self.train_loader):
            # Move data to device
            images = images.to(self.device)
            angles = angles.to(self.device)
            labels = labels.to(self.device).view(-1, 1)  # Ensure shape (B, 1)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, angles)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.

        Returns:
            float: Average validation loss (Log Loss).
        """
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, angles, labels in self.val_loader:
                images = images.to(self.device)
                angles = angles.to(self.device)
                labels = labels.to(self.device).view(-1, 1)

                outputs = self.model(images, angles)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def fit(self, save_path=None):
        """
        Runs the full training pipeline with Early Stopping.

        Args:
            save_path (str, optional): Path to save the best model weights.

        Returns:
            float: The best validation loss achieved.
        """
        logger.info(f"Starting training on device: {self.device}")
        logger.info(
            f"Configuration: Epochs={Config.EPOCHS}, LR={Config.LEARNING_RATE}, Batch={Config.BATCH_SIZE}"
        )

        early_stopping_counter = 0
        start_time = time.time()

        for epoch in range(1, Config.EPOCHS + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Scheduler Step
            # Note: ReduceLROnPlateau expects the metric (val_loss)
            self.scheduler.step(val_loss)

            # Get current LR for logging
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_duration = time.time() - epoch_start

            # Print metrics with full precision
            logger.info(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_duration:.2f}s"
            )

            # Early Stopping Logic
            if val_loss < self.best_val_loss:
                logger.info(
                    f"Validation loss improved from {self.best_val_loss:.8f} to {val_loss:.8f}. Saving model..."
                )
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0

                # Save to disk if path provided
                if save_path:
                    torch.save(self.best_model_state, save_path)
            else:
                early_stopping_counter += 1
                logger.info(
                    f"Validation loss did not improve. Counter: {early_stopping_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

        total_time = time.time() - start_time
        logger.info(
            f"Training completed in {total_time:.2f}s. Best Val Loss: {self.best_val_loss:.8f}"
        )

        # Load best weights into the model before returning
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return self.best_val_loss
