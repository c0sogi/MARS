import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
from library.config import Config
from library.utils import save_checkpoint


class Trainer:
    """
    Trainer class to manage the training and validation loops, including optimization,
    scheduling, and early stopping with weight preservation.
    """

    def __init__(self, model, device, optimizer, scheduler=None, criterion=None):
        """
        Args:
            model: The PyTorch model to train.
            device: The device (cpu/cuda) to run on.
            optimizer: The optimizer instance.
            scheduler: Learning rate scheduler (optional).
            criterion: Loss function. Defaults to BCEWithLogitsLoss if None.
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion if criterion is not None else nn.BCEWithLogitsLoss()
        self.best_model_state = None

    def train_one_epoch(self, train_loader):
        """
        Executes one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, angles, labels) in enumerate(train_loader):
            images = images.to(self.device)
            angles = angles.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # Ensure shape (B, 1)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, angles)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns:
            epoch_loss: The average BCE loss.
            metric_log_loss: The Log Loss metric (sklearn).
            metric_accuracy: The Accuracy metric.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for images, angles, labels in val_loader:
                images = images.to(self.device)
                angles = angles.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                outputs = self.model(images, angles)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)

                # Convert logits to probabilities for metrics
                probs = torch.sigmoid(outputs)

                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / len(val_loader.dataset)

        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)

        # Calculate Log Loss (Competition Metric)
        # Clip probabilities slightly to avoid log(0) if extreme values occur
        eps = 1e-15
        all_probs_clipped = np.clip(all_probs, eps, 1 - eps)
        metric_log_loss = log_loss(all_targets, all_probs_clipped, labels=[0, 1])

        # Calculate Accuracy
        predictions = (all_probs > 0.5).astype(int)
        metric_accuracy = accuracy_score(all_targets, predictions)

        return epoch_loss, metric_log_loss, metric_accuracy

    def fit(self, train_loader, val_loader, epochs, patience, save_path):
        """
        Runs the full training process with Early Stopping.

        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            epochs: Maximum number of epochs.
            patience: Number of epochs to wait for improvement before stopping.
            save_path: File path to save the best model checkpoint.

        Returns:
            best_val_loss: The best validation loss achieved.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(
            f"Starting training on {self.device} for {epochs} epochs with patience {patience}..."
        )

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_log_loss, val_acc = self.validate(val_loader)

            # Update Learning Rate
            if self.scheduler:
                self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Print metrics with full precision
            print(
                f"Epoch {epoch}: "
                f"Train Loss = {train_loss}, "
                f"Val Loss = {val_loss}, "
                f"Val Log Loss = {val_log_loss}, "
                f"Val Accuracy = {val_acc}, "
                f"LR = {current_lr}"
            )

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Deepcopy the model state to preserve the exact weights
                self.best_model_state = copy.deepcopy(self.model.state_dict())

                # Save checkpoint to disk
                save_checkpoint(self.best_model_state, save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss}"
                    )
                    break

        # Restore the best model weights into the model instance
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return best_val_loss
