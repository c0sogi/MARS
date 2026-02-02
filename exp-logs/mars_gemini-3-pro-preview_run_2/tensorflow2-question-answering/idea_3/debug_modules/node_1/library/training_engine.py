import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from library.config import Config


class TrainingEngine:
    """
    Encapsulates the training loop logic, including forward passes,
    loss calculation, optimization, and evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module = nn.BCEWithLogitsLoss(),
    ):
        """
        Args:
            model: The neural network model (SiameseGatedConvRanker).
            device: The torch device (CPU or GPU).
            optimizer: The optimizer (e.g., Adam).
            criterion: The loss function (default: BCEWithLogitsLoss).
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

    def train_one_epoch(self, train_loader: torch.utils.data.DataLoader) -> float:
        """
        Performs one epoch of training.

        Args:
            train_loader: DataLoader for the training set.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            q_idx = batch["q_indices"].to(self.device)
            c_idx = batch["c_indices"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Flatten batch if dimensions are [batch, num_samples, seq_len]
            # This handles the negative sampling structure
            if c_idx.dim() == 3:
                b, n, l = c_idx.shape
                # Repeat question for each candidate (positive + negatives)
                q_idx = q_idx.unsqueeze(1).expand(-1, n, -1).reshape(-1, q_idx.size(1))
                c_idx = c_idx.reshape(-1, l)
                labels = labels.reshape(-1)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(q_idx, c_idx)

            # Calculate loss
            loss = self.criterion(logits, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def evaluate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Evaluates the model on the validation set.

        Args:
            val_loader: DataLoader for the validation set.

        Returns:
            A tuple of (Average Validation Loss, Validation Accuracy).
        """
        self.model.eval()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                q_idx = batch["q_indices"].to(self.device)
                c_idx = batch["c_indices"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Flatten batch for pointwise evaluation
                if c_idx.dim() == 3:
                    b, n, l = c_idx.shape
                    q_idx = (
                        q_idx.unsqueeze(1).expand(-1, n, -1).reshape(-1, q_idx.size(1))
                    )
                    c_idx = c_idx.reshape(-1, l)
                    labels = labels.reshape(-1)

                # Forward pass
                logits = self.model(q_idx, c_idx)

                # Calculate loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1

                # Calculate accuracy
                # Sigmoid > 0.5 is equivalent to Logits > 0.0
                preds = (logits > 0.0).float()
                correct_preds += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_preds / total_samples if total_samples > 0 else 0.0

        return avg_loss, accuracy

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        patience: int,
        save_path: str,
    ):
        """
        Runs the full training loop with Early Stopping.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            num_epochs: Maximum number of epochs.
            patience: Early stopping patience.
            save_path: Path to save the best model.
        """
        print(f"Starting training for {num_epochs} epochs on {self.device}...")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Acc: {val_acc}"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"  Validation loss improved. Model saved to {save_path}")
            else:
                patience_counter += 1
                print(
                    f"  Validation loss did not improve. Patience: {patience_counter}/{patience}"
                )
                if patience_counter >= patience:
                    print("  Early stopping triggered.")
                    break
