import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import compute_mcc, seed_everything


class Trainer:
    """
    Trainer class for the TR-GCN model.
    Handles training loops, validation with threshold optimization,
    early stopping, and submission generation.
    """

    def __init__(self, model, optimizer, device=Config.DEVICE):
        """
        Args:
            model (nn.Module): The TR-GCN model.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (str): Device to run training on ('cpu' or 'cuda').
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # Initialize positive class weight for loss function
        # Using a tensor for efficient computation on device
        self.pos_weight = torch.tensor(Config.POS_WEIGHT, device=device)

        # Store the best threshold found during validation
        self.best_threshold = 0.5

    def weighted_bce_loss(self, y_pred, y_true):
        """
        Computes Weighted Binary Cross Entropy Loss.

        Args:
            y_pred (torch.Tensor): Predicted probabilities (Batch, 1).
            y_true (torch.Tensor): Ground truth labels (Batch, 1).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Clamp predictions to prevent log(0)
        y_pred = torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)

        # Weighted Loss Formula:
        # Loss = - [ (pos_weight * y * log(p)) + ((1-y) * log(1-p)) ]

        pos_term = self.pos_weight * y_true * torch.log(y_pred)
        neg_term = (1.0 - y_true) * torch.log(1.0 - y_pred)

        loss = -(pos_term + neg_term)

        return torch.mean(loss)

    def train_one_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device).unsqueeze(1)  # Shape (Batch, 1)

            self.optimizer.zero_grad()

            preds = self.model(batch_X)
            loss = self.weighted_bce_loss(preds, batch_y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate_one_epoch(self, val_loader):
        """
        Runs validation, computes loss, and finds the optimal decision threshold
        that maximizes MCC.

        Returns:
            tuple: (average_loss, best_mcc, best_threshold)
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device).unsqueeze(1)

                preds = self.model(batch_X)
                loss = self.weighted_bce_loss(preds, batch_y)

                total_loss += loss.item()
                num_batches += 1

                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Concatenate all predictions and targets
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Threshold Optimization
        # Grid search for the threshold that maximizes MCC on the validation set
        best_mcc = -1.0
        best_thresh = 0.5

        # Check thresholds from 0.1 to 0.9
        thresholds = np.arange(0.1, 0.91, 0.05)

        for t in thresholds:
            bin_preds = (all_preds >= t).astype(int)
            mcc = compute_mcc(all_targets, bin_preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = t

        return avg_loss, best_mcc, best_thresh

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS, patience=5):
        """
        Main training loop with Early Stopping.
        """
        best_val_mcc = -float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

        print(f"Starting training on device: {self.device}")
        print(f"Training for {epochs} epochs with patience {patience}")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_mcc, val_thresh = self.validate_one_epoch(val_loader)

            # Print metrics with full precision
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val MCC: {val_mcc}")
            print(f"Best Threshold: {val_thresh}")

            # Early Stopping Logic based on MCC
            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                self.best_threshold = val_thresh
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

        print(f"Training complete. Best Validation MCC: {best_val_mcc}")
        return best_val_mcc

    def generate_submission(self, test_loader, output_path=Config.SUBMISSION_PATH):
        """
        Generates predictions for the test set using the best model and threshold,
        and saves them to a CSV file.
        """
        print("Generating submission...")

        # Load best model weights
        best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model weights.")
        else:
            print("Warning: Best model not found. Using current model weights.")

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in test_loader:
                # Handle case where loader returns (X, y) or just X
                if isinstance(batch, (tuple, list)):
                    batch_X = batch[0]
                else:
                    batch_X = batch

                batch_X = batch_X.to(self.device)
                preds = self.model(batch_X)
                all_preds.append(preds.cpu().numpy())

        all_preds = np.concatenate(all_preds)

        # Apply the optimized threshold found during validation
        print(f"Applying optimized threshold: {self.best_threshold}")
        binary_preds = (all_preds >= self.best_threshold).astype(int)

        # Retrieve contact_ids from the dataset
        if hasattr(test_loader.dataset, "contact_ids"):
            contact_ids = test_loader.dataset.contact_ids
        else:
            raise AttributeError("Test dataset must have 'contact_ids' attribute.")

        if len(contact_ids) != len(binary_preds):
            raise ValueError(
                f"Shape mismatch: {len(contact_ids)} IDs vs {len(binary_preds)} predictions."
            )

        # Create submission DataFrame
        submission = pd.DataFrame(
            {"contact_id": contact_ids, "contact": binary_preds.flatten()}
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
