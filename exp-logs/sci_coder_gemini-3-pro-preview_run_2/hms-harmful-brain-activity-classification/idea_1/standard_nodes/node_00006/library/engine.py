import torch
import torch.nn as nn
import numpy as np
import os
import time
from library.config import Config


class Trainer:
    """
    Encapsulates the training, validation, and prediction loops for the EEG classification model.
    """

    def __init__(self, model, optimizer, device, scheduler=None, mixup_alpha=0.0):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            optimizer (torch.optim.Optimizer): The optimizer.
            device (str): Computing device ('cuda' or 'cpu').
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            mixup_alpha (float): Alpha parameter for Beta distribution in MixUp. 0.0 disables it.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.mixup_alpha = mixup_alpha

        # KL Divergence Loss
        # reduction='batchmean' mathematically aligns with the KL Divergence definition
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def train_one_epoch(self, data_loader, epoch_index):
        """
        Trains the model for one epoch.

        Args:
            data_loader (DataLoader): Training data loader.
            epoch_index (int): Current epoch number (for logging).

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, targets) in enumerate(data_loader):
            images = images.to(self.device, dtype=torch.float32)
            targets = targets.to(self.device, dtype=torch.float32)

            batch_size = images.size(0)

            # Zero gradients
            self.optimizer.zero_grad()

            # MixUp Regularization (Cite solution_lesson_node_00003)
            if self.mixup_alpha > 0 and self.model.training:
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                index = torch.randperm(batch_size).to(self.device)

                mixed_images = lam * images + (1 - lam) * images[index, :]
                mixed_targets = lam * targets + (1 - lam) * targets[index, :]

                # Forward pass with mixed images
                probs = self.model(mixed_images)
                log_probs = torch.log(probs + 1e-6)

                # Compute loss against mixed targets
                loss = self.criterion(log_probs, mixed_targets)
            else:
                # Standard Forward pass
                probs = self.model(images)
                log_probs = torch.log(probs + 1e-6)
                loss = self.criterion(log_probs, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            if Config.MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.MAX_GRAD_NORM
                )

            # Optimizer step
            self.optimizer.step()

            # Accumulate loss
            # loss.item() is the average loss per batch (due to reduction='batchmean')
            # We multiply by batch_size to accumulate total loss, then divide later
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate_one_epoch(self, data_loader):
        """
        Evaluates the model on the validation set.

        Args:
            data_loader (DataLoader): Validation data loader.

        Returns:
            float: Average validation loss (KL Divergence).
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        with torch.no_grad():
            for images, targets in data_loader:
                images = images.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32)

                batch_size = images.size(0)

                # Forward pass
                probs = self.model(images)
                log_probs = torch.log(probs + 1e-6)

                # Compute loss
                loss = self.criterion(log_probs, targets)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def fit(self, train_loader, val_loader, epochs, patience, save_path):
        """
        Main training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
            save_path (str): Path to save the best model weights.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss = self.validate_one_epoch(val_loader)

            # Scheduler Step
            if self.scheduler:
                # If scheduler is ReduceLROnPlateau, it needs the metric
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            duration = time.time() - start_time

            # Log Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Time: {duration:.2f}s"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"Validation loss improved. Model saved to {save_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Loss: {best_val_loss}")

    def predict(self, data_loader):
        """
        Generates predictions for a dataset.

        Args:
            data_loader (DataLoader): Test data loader.

        Returns:
            np.ndarray: Predicted probabilities of shape (N_samples, 6).
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in data_loader:
                # Handle cases where loader returns (images, targets) or just images
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch

                images = images.to(self.device, dtype=torch.float32)

                # Forward pass
                probs = self.model(images)

                # Move to CPU and store
                predictions.append(probs.cpu().numpy())

        return np.concatenate(predictions, axis=0)
