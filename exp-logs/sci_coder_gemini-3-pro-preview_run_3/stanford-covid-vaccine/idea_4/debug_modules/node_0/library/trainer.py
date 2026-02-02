import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import MCRMSELoss, compute_mcrmse
from library.model import RNAConformer


class Trainer:
    """
    Trainer class for the RNA Conformer model.
    Handles the training loop, validation, checkpointing, and early stopping.
    """

    def __init__(self, model=None, device=None):
        """
        Initialize the Trainer.

        Args:
            model (nn.Module, optional): The model to train. If None, instantiates RNAConformer.
            device (str, optional): Device to run on. If None, uses Config.DEVICE.
        """
        self.config = Config
        self.device = device if device else self.config.DEVICE

        # Initialize model
        self.model = model if model else RNAConformer()
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.EPOCHS
        )

        # Loss Function
        self.criterion = MCRMSELoss()

        # State tracking
        self.best_score = float("inf")
        self.best_model_path = os.path.join(self.config.WORKING_DIR, "best_model.pth")

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.

        Args:
            dataloader (DataLoader): Training data loader.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            inputs = batch["inputs"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(inputs)

            # Slice to scored length (first 68 positions)
            # targets are padded with 0s after 68, but we only want to score valid positions
            preds_scored = preds[:, : self.config.SCORED_LENGTH, :]
            targets_scored = targets[:, : self.config.SCORED_LENGTH, :]

            # Compute loss
            loss = self.criterion(preds_scored, targets_scored)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        return running_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, dataloader):
        """
        Runs validation on the given dataloader.

        Args:
            dataloader (DataLoader): Validation data loader.

        Returns:
            float: MCRMSE score on the validation set.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["inputs"].to(self.device)
                targets = batch["targets"].to(self.device)

                # Forward pass
                preds = self.model(inputs)

                # Slice to scored length
                preds_scored = preds[:, : self.config.SCORED_LENGTH, :]
                targets_scored = targets[:, : self.config.SCORED_LENGTH, :]

                all_preds.append(preds_scored.cpu())
                all_targets.append(targets_scored.cpu())

        # Concatenate all batches
        if not all_preds:
            return 0.0

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute metric using the utility function
        score = compute_mcrmse(all_preds, all_targets)
        return score

    def fit(self, train_loader, val_loader):
        """
        Main training loop with early stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
        """
        print(f"Starting training on device: {self.device}")
        print(f"Epochs: {self.config.EPOCHS}, Batch Size: {self.config.BATCH_SIZE}")

        patience_counter = 0

        for epoch in range(1, self.config.EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_score = self.validate(val_loader)

            # Update Scheduler
            self.scheduler.step()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss} | "
                f"Val MCRMSE: {val_score} | "
                f"Time: {elapsed:.2f}s"
            )

            # Checkpointing and Early Stopping
            if val_score < self.best_score:
                self.best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  [New Best Model] Saved to {self.best_model_path}")
            else:
                patience_counter += 1
                print(f"  [Patience] {patience_counter}/{self.config.PATIENCE}")

            if patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation MCRMSE: {self.best_score}")

    def load_best_model(self):
        """
        Loads the best model state from the checkpoint.
        """
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print(f"Loaded best model from {self.best_model_path}")
        else:
            print(f"Warning: No checkpoint found at {self.best_model_path}")

    def predict(self, dataloader):
        """
        Generates predictions for a dataset.

        Args:
            dataloader (DataLoader): Test data loader.

        Returns:
            np.ndarray: Predictions of shape (N, Seq_Len, 5).
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["inputs"].to(self.device)

                # Forward pass
                preds = self.model(inputs)

                # Keep full sequence length (107) for submission format requirements
                # The submission utility handles flattening, but expects (N, 107, 5)
                all_preds.append(preds.cpu())

        if not all_preds:
            return np.array([])

        return torch.cat(all_preds, dim=0).numpy()
