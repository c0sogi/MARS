import os
import time
import numpy as np
import torch
import torch.nn as nn
from library.config import Config
from library.utils import AverageMeter, kl_divergence_score


class Trainer:
    """
    Manages the training, validation, and checkpointing process for the
    Pyramid-Resolution Coordinate-Guided Fusion Network.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        patience=Config.PATIENCE,
    ):
        """
        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (Optimizer): PyTorch optimizer.
            scheduler (LRScheduler): PyTorch learning rate scheduler.
            device (str): Device to run training on ('cuda' or 'cpu').
            patience (int): Epochs to wait for improvement before early stopping.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience

        # Loss function: KLDivLoss expects log-probabilities as input
        self.criterion = nn.KLDivLoss(reduction="batchmean")

        self.best_val_score = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_score": []}

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        loss_meter = AverageMeter()

        for batch_idx, (eeg, spec, targets) in enumerate(self.train_loader):
            eeg = eeg.to(self.device, non_blocking=True)
            spec = spec.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Forward pass
            # Model outputs Softmax probabilities
            outputs = self.model(eeg, spec)

            # KLDivLoss requires Log-Softmax inputs
            # Add epsilon for numerical stability inside log just in case,
            # though model output is softmax so it should be > 0.
            log_outputs = torch.log(outputs + 1e-15)

            loss = self.criterion(log_outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if Config.MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.MAX_GRAD_NORM
                )

            self.optimizer.step()

            # Step scheduler (OneCycleLR steps per batch)
            if self.scheduler is not None:
                self.scheduler.step()

            loss_meter.update(loss.item(), eeg.size(0))

        return loss_meter.avg

    def validate(self):
        """
        Runs validation on the validation set and computes the KL Divergence score.
        """
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for eeg, spec, targets in self.val_loader:
                eeg = eeg.to(self.device, non_blocking=True)
                spec = spec.to(self.device, non_blocking=True)

                # Forward pass
                outputs = self.model(eeg, spec)

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate Metric using the provided utility
        score = kl_divergence_score(all_targets, all_preds)
        return score

    def fit(self, epochs):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_score = self.validate()

            elapsed = time.time() - start_time

            # Log metrics
            print(
                f"Epoch {epoch}/{epochs} | "
                f"Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val KL Score: {val_score:.16f}"
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_score"].append(val_score)

            # Checkpointing & Early Stopping
            if val_score < self.best_val_score:
                print(
                    f"Validation score improved ({self.best_val_score:.8f} --> {val_score:.8f}). Saving model..."
                )
                self.best_val_score = val_score
                self.patience_counter = 0
                self.save_checkpoint("best_model.pth")
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{self.patience}"
                )

            if self.patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

    def save_checkpoint(self, filename):
        """
        Saves the model state to the checkpoint directory.
        """
        save_path = os.path.join(Config.CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
