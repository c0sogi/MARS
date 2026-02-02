import time
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import set_seed, calculate_lwlrap, save_checkpoint


class Trainer:
    """
    Trainer class to handle model training, validation, and checkpointing.
    """

    def __init__(self, model, train_loader, val_loader, optimizer, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = Config.device
        self.criterion = nn.BCEWithLogitsLoss()

        # Initialize early stopping tracking
        self.patience_counter = 0
        if Config.early_stopping_mode == "max":
            self.best_score = -np.inf
        else:
            self.best_score = np.inf

        set_seed(Config.seed)

    def train_one_epoch(self, epoch_index):
        """
        Runs one epoch of training.
        Applies Mixup augmentation and updates model weights.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            batch_size = images.size(0)

            # Apply Mixup if configured
            if Config.mixup_alpha > 0:
                # Sample lambda from Beta distribution
                lam = np.random.beta(Config.mixup_alpha, Config.mixup_alpha)

                # Shuffle indices for mixing
                index = torch.randperm(batch_size).to(self.device)

                # Mix inputs
                mixed_images = lam * images + (1 - lam) * images[index]

                # Mix targets (BCEWithLogitsLoss supports soft labels)
                mixed_targets = lam * targets + (1 - lam) * targets[index]

                # Forward pass
                outputs = self.model(mixed_images)
                loss = self.criterion(outputs, mixed_targets)
            else:
                # Standard training without Mixup
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Step scheduler (OneCycleLR requires stepping every batch)
            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self):
        """
        Runs validation on the validation set.
        Computes Loss and LWLRAP.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                batch_size = images.size(0)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Convert logits to probabilities for metric calculation
                preds = torch.sigmoid(outputs)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        val_loss = running_loss / dataset_size

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate metric
        val_lrap = calculate_lwlrap(all_targets, all_preds)

        return val_loss, val_lrap

    def fit(self):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")
        print(f"Epochs: {Config.epochs}, Batch Size: {Config.batch_size}")
        print(
            f"Early Stopping Metric: {Config.early_stopping_metric}, Patience: {Config.early_stopping_patience}"
        )

        for epoch in range(1, Config.epochs + 1):
            start_time = time.time()

            # Train and Validate
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_lrap = self.validate()

            elapsed = time.time() - start_time

            # Print metrics
            print(
                f"Epoch {epoch}/{Config.epochs} - "
                f"Time: {elapsed:.2f}s - "
                f"Train Loss: {train_loss:.8f} - "
                f"Val Loss: {val_loss:.8f} - "
                f"Val LWLRAP: {val_lrap}"
            )

            # Determine current score based on configured metric
            if Config.early_stopping_metric == "lrap":
                current_score = val_lrap
            else:
                current_score = val_loss

            # Check for improvement
            improved = False
            if Config.early_stopping_mode == "max":
                if current_score > self.best_score:
                    improved = True
            else:
                if current_score < self.best_score:
                    improved = True

            if improved:
                self.best_score = current_score
                self.patience_counter = 0
                print(
                    f"New best score ({current_score})! Saving checkpoint to {Config.checkpoint_path}"
                )
                save_checkpoint(self.model, Config.checkpoint_path)
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{Config.early_stopping_patience}"
                )

            if self.patience_counter >= Config.early_stopping_patience:
                print("Early stopping triggered.")
                break
