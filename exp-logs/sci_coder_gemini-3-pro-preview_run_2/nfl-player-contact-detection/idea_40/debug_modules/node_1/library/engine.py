import torch
import torch.nn as nn
import numpy as np
import os
from library.config import Config
from library.loss import FocalLoss
from library.utils import compute_mcc


class LRPNetEngine:
    """
    Engine class to handle training, evaluation, and threshold optimization
    for the LRP-Net model.
    """

    def __init__(self, model, device, optimizer=None, scheduler=None):
        """
        Args:
            model (torch.nn.Module): The LRP-Net model.
            device (str): Device to run training on ('cpu' or 'cuda').
            optimizer (torch.optim.Optimizer, optional): Optimizer.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)

        # Early Stopping State
        self.best_mcc = -1.0
        self.best_threshold = 0.5
        self.patience_counter = 0

    def train_one_epoch(self, train_loader):
        """
        Performs one epoch of training.

        Args:
            train_loader (DataLoader): Loader for training data.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device).unsqueeze(1)  # Ensure shape [B, 1]

            if self.optimizer:
                self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(data)
            loss = self.criterion(logits, target)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            if self.optimizer:
                self.optimizer.step()

            running_loss += loss.item() * data.size(0)
            count += data.size(0)

        epoch_loss = running_loss / count if count > 0 else 0.0
        return epoch_loss

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.

        Args:
            val_loader (DataLoader): Loader for validation data.

        Returns:
            tuple: (y_probs, y_true) as numpy arrays.
                   y_probs are probabilities (after sigmoid).
        """
        self.model.eval()
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)

                # Forward pass
                logits = self.model(data)
                probs = torch.sigmoid(logits)

                preds_list.append(probs.cpu().numpy())
                targets_list.append(target.numpy())

        # Concatenate all batches
        y_probs = np.concatenate(preds_list).flatten()
        y_true = np.concatenate(targets_list).flatten()

        return y_probs, y_true

    def optimize_threshold(self, y_true, y_probs):
        """
        Performs a grid search to find the classification threshold that maximizes MCC.

        Args:
            y_true (np.array): Ground truth labels.
            y_probs (np.array): Predicted probabilities.

        Returns:
            tuple: (best_threshold, best_mcc)
        """
        thresholds = np.arange(0.01, 1.00, 0.01)
        best_mcc = -1.0
        best_thresh = 0.5

        # Vectorized thresholding check is memory intensive for large arrays,
        # so we loop. The overhead is minimal for 100 steps.
        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            score = compute_mcc(y_true, y_pred)

            if score > best_mcc:
                best_mcc = score
                best_thresh = thresh

        return best_thresh, best_mcc

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS):
        """
        Runs the full training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_one_epoch(train_loader)

            # Validate
            val_probs, val_true = self.evaluate(val_loader)

            # Optimize Threshold on Validation Set
            # Note: In a strict setup, threshold should be tuned on a separate calibration set,
            # but standard competition practice often uses the validation set for this.
            curr_thresh, curr_mcc = self.optimize_threshold(val_true, val_probs)

            # Print metrics
            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Val MCC: {curr_mcc} | Best Thresh: {curr_thresh}"
            )

            # Scheduler Step
            if self.scheduler:
                self.scheduler.step(curr_mcc)

            # Early Stopping Check
            if curr_mcc > self.best_mcc:
                self.best_mcc = curr_mcc
                self.best_threshold = curr_thresh
                self.patience_counter = 0

                # Save Best Model
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                # Save Best Threshold
                np.save(
                    os.path.join(Config.WORKING_DIR, "best_threshold.npy"),
                    np.array([self.best_threshold]),
                )

            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered at epoch {epoch}. Best Val MCC: {self.best_mcc}"
                    )
                    break

        # Load best model weights before exiting
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
            print("Loaded best model weights.")

    def predict(self, test_loader, threshold=None):
        """
        Generates binary predictions for a test set.

        Args:
            test_loader (DataLoader): Test data.
            threshold (float, optional): Threshold to use. If None, uses self.best_threshold.

        Returns:
            np.array: Binary predictions.
        """
        self.model.eval()
        preds_list = []

        thresh = threshold if threshold is not None else self.best_threshold

        with torch.no_grad():
            for data in test_loader:
                # Handle case where loader returns (data, target) or just data
                if isinstance(data, (list, tuple)):
                    data = data[0]

                data = data.to(self.device)
                logits = self.model(data)
                probs = torch.sigmoid(logits)
                preds_list.append(probs.cpu().numpy())

        all_probs = np.concatenate(preds_list).flatten()
        predictions = (all_probs >= thresh).astype(int)

        return predictions
