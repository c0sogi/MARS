import torch
import numpy as np
import os
from library.config import Config
from library.utils import compute_mcc
from library.loss import FocalLoss


class Trainer:
    """
    Manages the training, validation, and threshold optimization for the EF-WideResNet model.
    """

    def __init__(
        self, model, train_loader, val_loader, optimizer, device=Config.DEVICE
    ):
        """
        Args:
            model (torch.nn.Module): The EF-WideResNet model.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (torch.optim.Optimizer): Optimizer (e.g., AdamW).
            device (str): Device to run training on ('cpu' or 'cuda').
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device

        # Initialize Focal Loss
        self.criterion = FocalLoss()

        # Move model to the specified device
        self.model.to(self.device)

    def train_one_epoch(self):
        """
        Runs one epoch of training.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Unpack batch: continuous features, categorical dict, labels
            x_cont, x_cat, labels = batch

            # Move data to device
            x_cont = x_cont.to(self.device)
            labels = labels.to(self.device)
            # Handle dictionary of categorical tensors
            x_cat = {k: v.to(self.device) for k, v in x_cat.items()}

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(x_cont, x_cat)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        """
        Evaluates the model on the validation set.

        Returns:
            tuple: (average_loss, all_probabilities, all_targets)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                x_cont, x_cat, labels = batch

                x_cont = x_cont.to(self.device)
                labels = labels.to(self.device)
                x_cat = {k: v.to(self.device) for k, v in x_cat.items()}

                outputs = self.model(x_cont, x_cat)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                num_batches += 1

                # Apply sigmoid to convert logits to probabilities
                probs = torch.sigmoid(outputs)

                all_probs.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Concatenate predictions from all batches
        if all_probs:
            all_probs = np.concatenate(all_probs)
            all_targets = np.concatenate(all_targets)
        else:
            all_probs = np.array([])
            all_targets = np.array([])

        return avg_loss, all_probs, all_targets

    def optimize_threshold(self, y_true, y_probs):
        """
        Performs a grid search to find the probability threshold that maximizes MCC.

        Args:
            y_true (np.ndarray): Ground truth binary labels.
            y_probs (np.ndarray): Predicted probabilities.

        Returns:
            tuple: (best_threshold, best_mcc_score)
        """
        best_mcc = -1.0
        best_thresh = 0.5

        # Search thresholds from 0.01 to 0.99
        thresholds = np.arange(0.01, 1.00, 0.01)

        # Flatten arrays to ensure 1D shape
        y_true = y_true.flatten()
        y_probs = y_probs.flatten()

        for thresh in thresholds:
            # Binarize predictions based on current threshold
            y_pred = (y_probs >= thresh).astype(int)

            # Compute MCC
            mcc = compute_mcc(y_true, y_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = thresh

        return best_thresh, best_mcc

    def fit(
        self,
        epochs=Config.EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
        save_path=Config.MODEL_PATH,
    ):
        """
        Runs the full training loop with Early Stopping.

        Args:
            epochs (int): Maximum number of epochs.
            patience (int): Number of epochs to wait for improvement before stopping.
            save_path (str): Path to save the best model weights.

        Returns:
            float: The best threshold found during training.
        """
        best_val_mcc = -1.0
        patience_counter = 0
        best_threshold = 0.5

        print(f"Starting training for {epochs} epochs on device: {self.device}")

        for epoch in range(1, epochs + 1):
            # 1. Train
            train_loss = self.train_one_epoch()

            # 2. Validate
            val_loss, val_probs, val_targets = self.validate()

            # 3. Optimize Threshold for current epoch
            # This ensures we are monitoring the best possible performance of the model
            curr_thresh, curr_mcc = self.optimize_threshold(val_targets, val_probs)

            # Print full precision metrics
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.10f} | "
                f"Val Loss: {val_loss:.10f} | "
                f"Val MCC: {curr_mcc:.10f} | "
                f"Best Thresh: {curr_thresh:.2f}"
            )

            # 4. Early Stopping Check
            if curr_mcc > best_val_mcc:
                best_val_mcc = curr_mcc
                best_threshold = curr_thresh
                patience_counter = 0

                # Save the best model
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved to {save_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(
            f"Training complete. Best Val MCC: {best_val_mcc:.10f} at Threshold: {best_threshold:.2f}"
        )
        return best_threshold
