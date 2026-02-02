import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from library.config import Config
from library.utils import mcrmse


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        """
        Initializes the Trainer.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = Config.DEVICE

        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # Loss Function
        # We use MSELoss as a stable proxy for MCRMSE during optimization.
        # It is calculated on all 5 targets.
        self.criterion = nn.MSELoss()

        # Early Stopping parameters
        self.best_score = float("inf")
        self.patience_counter = 0
        self.patience = Config.PATIENCE
        self.model_path = Config.MODEL_PATH

    def train_one_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch in self.train_loader:
            # Move data to device
            inputs = batch["input"].to(self.device)
            adjacency = batch["adjacency"].to(self.device)
            pair_mask = batch["pair_mask"].to(self.device)
            targets = batch["target"].to(self.device)  # Shape: (B, 68, 5)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs, adjacency, pair_mask)  # Shape: (B, 107, 5)

            # Slice outputs to match targets (first 68 positions)
            # Targets are provided for the first 68 positions (seq_scored)
            outputs_sliced = outputs[:, : Config.PRED_LEN, :]

            # Compute loss
            loss = self.criterion(outputs_sliced, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            # Optimization step
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss

    def validate(self):
        """
        Runs validation and calculates MCRMSE on scored columns.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["input"].to(self.device)
                adjacency = batch["adjacency"].to(self.device)
                pair_mask = batch["pair_mask"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = self.model(inputs, adjacency, pair_mask)

                # Store predictions and targets for global metric calculation
                # We keep them on CPU to save GPU memory
                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate MCRMSE using the utility function
        # This handles slicing and filtering for scored columns internally
        score = mcrmse(all_targets, all_preds, only_scored=True)

        return score.item()

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_one_epoch()
            val_score = self.validate()

            # Step the scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss (MSE): {train_loss:.6f} | "
                f"Val MCRMSE: {val_score:.10f}"
            )

            # Early Stopping Logic
            if val_score < self.best_score:
                self.best_score = val_score
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.model_path)
                print(f"  New best model saved! Score: {self.best_score:.10f}")
            else:
                self.patience_counter += 1
                print(
                    f"  No improvement. Patience: {self.patience_counter}/{self.patience}"
                )

                if self.patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Val MCRMSE: {self.best_score:.10f}")
