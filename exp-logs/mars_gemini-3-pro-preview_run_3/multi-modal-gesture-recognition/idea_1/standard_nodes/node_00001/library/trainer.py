import torch
import torch.nn as nn
import torch.optim as optim
import os
from library.config import Config
from library.utils import setup_logger


class Trainer:
    """
    Trainer class to manage the training and validation of the BiGRUModel.
    """

    def __init__(self, model, train_loader, val_loader, config=Config):
        """
        Args:
            model (nn.Module): The BiGRU model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            config (class): Configuration class with hyperparameters.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.logger = setup_logger("Trainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Setup Loss Function with Class Weights
        # Background class (0) is very frequent, so we down-weight it.
        if self.config.USE_CLASS_WEIGHTS:
            # Weight 0.1 for background, 1.0 for all other 20 gesture classes
            weights = torch.ones(self.config.NUM_CLASSES)
            weights[self.config.BACKGROUND_CLASS_ID] = 0.1
            self.criterion = nn.CrossEntropyLoss(
                weight=weights.to(self.device), ignore_index=-100
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config.LEARNING_RATE
        )

    def train_epoch(self):
        """
        Runs one epoch of training.
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for features, labels, lengths in self.train_loader:
            # Move data to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            # lengths usually need to be on CPU for pack_padded_sequence in some PyTorch versions,
            # but model.forward handles .cpu() call internally if needed, or we pass as is.
            # library/model.py explicitly calls .cpu() on lengths, so we can pass tensor.

            self.optimizer.zero_grad()

            # Forward pass
            # logits: (Batch, MaxLen, NumClasses)
            logits = self.model(features, lengths)

            # Flatten outputs and targets for CrossEntropyLoss
            # logits -> (Batch * MaxLen, NumClasses)
            # labels -> (Batch * MaxLen)
            loss = self.criterion(
                logits.view(-1, self.config.NUM_CLASSES), labels.view(-1)
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self):
        """
        Runs validation on the validation set.
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for features, labels, lengths in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(features, lengths)

                loss = self.criterion(
                    logits.view(-1, self.config.NUM_CLASSES), labels.view(-1)
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(self, num_epochs=None):
        """
        Main training loop with Early Stopping.

        Args:
            num_epochs (int, optional): Override number of epochs from config.
        """
        epochs = num_epochs if num_epochs is not None else self.config.NUM_EPOCHS

        best_val_loss = float("inf")
        patience_counter = 0

        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(f"Total Epochs: {epochs}")

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Print metrics with full precision
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            # Checkpoint and Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                self.logger.info(
                    f"New best model saved to {self.config.MODEL_SAVE_PATH}"
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    self.logger.info(
                        f"Early stopping triggered after {patience_counter} epochs of no improvement."
                    )
                    break

        self.logger.info("Training complete.")
