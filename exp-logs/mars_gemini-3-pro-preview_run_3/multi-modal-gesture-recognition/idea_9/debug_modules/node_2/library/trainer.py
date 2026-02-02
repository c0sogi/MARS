import os
import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import setup_logger, set_seed
from library.model import VIARN
from library.loss import CascadedLoss
from library.data_processing import get_dataloaders


class Trainer:
    """
    Manages the training, validation, and model saving for the VI-ARN model.
    """

    def __init__(self):
        # Set deterministic behavior
        set_seed(Config.SEED)

        # Setup Logger
        self.logger = setup_logger(name="Trainer")

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Initialize Model
        self.model = VIARN().to(self.device)

        # Initialize Loss
        self.criterion = CascadedLoss()

        # Initialize Optimizer
        # Using Adam as specified in the design
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Training State
        self.start_epoch = 0
        self.best_val_loss = float("inf")

    def calculate_accuracy(self, probs, targets):
        """
        Computes frame-wise accuracy.
        probs: (Batch, Time, Classes)
        targets: (Batch, Time)
        """
        # Get predictions: argmax over class dimension
        preds = torch.argmax(probs, dim=2)  # (Batch, Time)

        # Mask out padding if necessary?
        # The dataloader pads with 0 (background) and targets are 0.
        # Accuracy on background is relevant, so we include it.
        correct = (preds == targets).float()
        acc = correct.sum() / correct.numel()
        return acc.item()

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()

        running_metrics = {}
        batch_count = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # Returns dict: {'stage1': ..., 'stage2': ..., 'stage3': ...}
            outputs = self.model(inputs)

            # Compute Loss
            loss, metrics = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Accumulate metrics
            # Initialize keys if first batch
            if batch_count == 0:
                for k in metrics:
                    running_metrics[k] = 0.0
                running_metrics["accuracy"] = 0.0

            for k, v in metrics.items():
                running_metrics[k] += v

            # Calculate accuracy based on final stage (stage3)
            acc = self.calculate_accuracy(outputs["stage3"], targets)
            running_metrics["accuracy"] += acc

            batch_count += 1

        # Average metrics
        avg_metrics = {k: v / batch_count for k, v in running_metrics.items()}
        return avg_metrics

    def validate(self, dataloader):
        """
        Runs validation on the provided dataloader.
        """
        self.model.eval()

        running_metrics = {}
        batch_count = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute Loss
                _, metrics = self.criterion(outputs, targets)

                # Accumulate metrics
                if batch_count == 0:
                    for k in metrics:
                        running_metrics[k] = 0.0
                    running_metrics["accuracy"] = 0.0

                for k, v in metrics.items():
                    running_metrics[k] += v

                # Calculate accuracy based on final stage
                acc = self.calculate_accuracy(outputs["stage3"], targets)
                running_metrics["accuracy"] += acc

                batch_count += 1

        if batch_count == 0:
            return {}

        avg_metrics = {k: v / batch_count for k, v in running_metrics.items()}
        return avg_metrics

    def fit(self, debug_subset=None):
        """
        Main training loop with Early Stopping.
        """
        self.logger.info("Starting training process...")

        # Load Data
        train_loader, val_loader, _, _ = get_dataloaders(debug_subset=debug_subset)

        # Early Stopping variables
        patience = Config.EARLY_STOPPING_PATIENCE
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            current_epoch = epoch + 1

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Logging
            # Print full precision as requested
            log_msg = f"Epoch {current_epoch}/{Config.NUM_EPOCHS} | "
            log_msg += f"Train Loss: {train_metrics['total_loss']} | "
            log_msg += f"Val Loss: {val_metrics['total_loss']} | "
            log_msg += f"Train Acc: {train_metrics['accuracy']} | "
            log_msg += f"Val Acc: {val_metrics['accuracy']}"
            self.logger.info(log_msg)

            # Checkpoint & Early Stopping
            val_loss = val_metrics["total_loss"]

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.logger.info(
                    f"Validation loss improved. Saving model to {Config.MODEL_SAVE_PATH}"
                )
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info("Training complete.")
