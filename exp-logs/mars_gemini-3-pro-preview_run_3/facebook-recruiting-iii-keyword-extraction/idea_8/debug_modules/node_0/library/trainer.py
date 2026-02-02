import torch
import torch.nn as nn
import numpy as np
import os
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import save_checkpoint, optimize_threshold


class ModelTrainer:
    """
    Manages the training, validation, and prediction processes for the Wide-and-Deep TextCNN.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler=None,
        device=Config.DEVICE,
    ):
        """
        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            criterion (nn.Module): Loss function (e.g., FocalLoss).
            optimizer (Optimizer): Optimizer (e.g., AdamW).
            scheduler (LRScheduler, optional): Learning rate scheduler (e.g., OneCycleLR).
            device (str): Device to run training on ('cuda' or 'cpu').
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scaler = GradScaler()  # For Automatic Mixed Precision

        self.model.to(self.device)

    def train_epoch(self):
        """
        Runs one epoch of training.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast():
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

            # Backward Pass with Scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Step Scheduler (OneCycleLR steps per batch)
            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / num_batches
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.

        Returns:
            tuple: (avg_loss, best_f1, best_threshold)
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with autocast():
                    logits = self.model(inputs)
                    loss = self.criterion(logits, targets)

                running_loss += loss.item()

                # Apply Sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)

        # Concatenate all batches
        y_true = np.vstack(all_targets)
        y_probs = np.vstack(all_probs)

        # Calculate F1-Score using threshold optimization
        # This prints the best threshold and score to stdout
        best_threshold, best_f1 = optimize_threshold(y_true, y_probs)

        return avg_loss, best_f1, best_threshold

    def fit(self, num_epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE):
        """
        Runs the full training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")
        best_val_f1 = -1.0
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            # Train
            train_loss = self.train_epoch()
            print(f"Training Loss: {train_loss}")

            # Validate
            val_loss, val_f1, val_thresh = self.validate()
            print(f"Validation Loss: {val_loss}")
            print(f"Validation F1: {val_f1}")

            # Checkpoint & Early Stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0

                checkpoint = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_f1": best_val_f1,
                    "best_threshold": val_thresh,
                }
                save_checkpoint(checkpoint, Config.MODEL_PATH)
                print(f"New best model saved with F1: {best_val_f1}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation F1: {best_val_f1}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set using the trained model.

        Args:
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            np.array: Predicted probabilities (N, NumTags).
        """
        self.model.eval()
        all_probs = []

        print("Generating predictions on test set...")
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device, non_blocking=True)

                with autocast():
                    logits = self.model(inputs)

                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)
