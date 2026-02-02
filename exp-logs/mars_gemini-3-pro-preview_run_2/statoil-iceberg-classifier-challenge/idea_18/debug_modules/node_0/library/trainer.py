import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.utils import get_logger, EarlyStopping


class ModelTrainer:
    """
    Manages the training and validation lifecycle for the WB-DSN model.
    """

    def __init__(self, model, device, logger=None, learning_rate=2e-4):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            device (torch.device): Device to run training on.
            logger (logging.Logger, optional): Logger instance.
            learning_rate (float): Learning rate for Adam optimizer.
        """
        self.model = model
        self.device = device
        self.logger = logger if logger else get_logger("ModelTrainer")

        # Criterion: Binary Cross Entropy with Logits (since model outputs logits)
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer: Adam with "Low and Slow" strategy
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Scheduler: ReduceLROnPlateau
        # Decays LR when validation loss stagnates
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.

        Args:
            train_loader (DataLoader): DataLoader for training data.

        Returns:
            tuple: (epoch_loss, epoch_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Unpack inputs: (images, angles)
            images, angles = inputs

            # Move to device
            images = images.to(self.device)
            angles = angles.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, angles)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size

            # Accuracy (Sigmoid > 0.5)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == targets).sum().item()
            total += batch_size

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """
        Runs validation.

        Args:
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            tuple: (val_loss, val_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                images, angles = inputs

                images = images.to(self.device)
                angles = angles.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images, angles)
                loss = self.criterion(outputs, targets)

                batch_size = targets.size(0)
                running_loss += loss.item() * batch_size

                preds = torch.sigmoid(outputs) > 0.5
                correct += (preds == targets).sum().item()
                total += batch_size

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self, train_loader, val_loader, epochs=50, patience=10):
        """
        Runs the full training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
            patience (int): Patience for early stopping.

        Returns:
            float: Best validation loss achieved.
        """
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        for epoch in range(epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Update scheduler based on validation loss
            self.scheduler.step(val_loss)

            elapsed = time.time() - start_time

            # Print metrics (Full precision for Val Loss as requested)
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} - Time: {elapsed:.2f}s - "
                f"Train Loss: {train_loss:.6f} - Train Acc: {train_acc:.6f} - "
                f"Val Loss: {val_loss} - Val Acc: {val_acc:.6f}"
            )

            # Check Early Stopping
            early_stopping(val_loss, self.model)

            if early_stopping.early_stop:
                self.logger.info("Early stopping triggered")
                break

        # Restore best weights
        early_stopping.restore_best_weights(self.model)
        return early_stopping.best_score

    def predict(self, test_loader):
        """
        Generates predictions for the test set.

        Args:
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            np.array: Flattened array of probabilities (0-1).
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs in test_loader:
                # Test loader inputs are (images, angles)
                images, angles = inputs

                images = images.to(self.device)
                angles = angles.to(self.device)

                outputs = self.model(images, angles)
                probs = torch.sigmoid(outputs)
                predictions.extend(probs.cpu().numpy().flatten())

        return np.array(predictions)
