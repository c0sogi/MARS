import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import set_seed, get_device, compute_metric
from library.data import get_dataloaders, get_test_loader
from library.model import MultiScaleResidualLSTM


class MaskedL1Loss(nn.Module):
    """
    Computes L1 Loss strictly for the inspiratory phase.
    Masks out time steps where u_out == 1 (expiratory phase).
    """

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, preds, targets, u_out):
        """
        Args:
            preds: Predicted pressure (Batch, Seq, 1) or (Batch, Seq)
            targets: Actual pressure (Batch, Seq)
            u_out: Control input (Batch, Seq), 0 for inspiratory, 1 for expiratory
        """
        # Ensure shapes match
        if preds.dim() == 3:
            preds = preds.squeeze(-1)
        if targets.dim() == 3:
            targets = targets.squeeze(-1)

        # Calculate element-wise L1 loss
        loss = self.l1(preds, targets)

        # Create mask: 1 where u_out == 0 (inspiratory), 0 otherwise
        mask = 1 - u_out

        # Apply mask
        masked_loss = loss * mask

        # Average over valid elements (add epsilon to avoid division by zero)
        return masked_loss.sum() / (mask.sum() + 1e-8)


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the model.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = get_device()

        # Ensure reproducibility
        set_seed(config.SEED)

        # Initialize Model
        self.model = MultiScaleResidualLSTM(config).to(self.device)

        # Loss Function
        self.criterion = MaskedL1Loss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler placeholder (initialized in fit() once loader length is known)
        self.scheduler = None

    def train_epoch(self, loader):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        for X, y, u_out in loader:
            X = X.to(self.device)
            y = y.to(self.device)
            u_out = u_out.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(X)

            # Compute Loss
            loss = self.criterion(preds, y, u_out)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            if self.config.CLIP_GRAD > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.CLIP_GRAD
                )

            # Update weights
            self.optimizer.step()

            # Update scheduler (OneCycleLR steps per batch)
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            total_steps += 1

        return total_loss / total_steps

    def validate(self, loader):
        """Evaluates the model on the validation set using the competition metric."""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_u_out = []

        with torch.no_grad():
            for X, y, u_out in loader:
                X = X.to(self.device)

                preds = self.model(X)

                # Move to CPU for metric calculation
                all_preds.append(preds.squeeze(-1).cpu().numpy())
                all_targets.append(y.cpu().numpy())
                all_u_out.append(u_out.cpu().numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_u_out = np.concatenate(all_u_out)

        # Calculate MAE using the utility function (handles masking)
        mae = compute_metric(all_preds, all_targets, all_u_out)
        return mae

    def fit(self, load_cached_data=True):
        """
        Main training loop.

        Args:
            load_cached_data (bool): Whether to use cached numpy files for data.
        """
        print(f"Initializing training on device: {self.device}")

        # Get DataLoaders
        train_loader, val_loader = get_dataloaders(self.config, load_cached_data)

        # Initialize OneCycleLR Scheduler
        steps_per_epoch = len(train_loader)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.LEARNING_RATE,
            epochs=self.config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            pct_start=self.config.PCT_START,
            div_factor=self.config.DIV_FACTOR,
            final_div_factor=self.config.FINAL_DIV_FACTOR,
        )

        best_mae = float("inf")

        for epoch in range(1, self.config.EPOCHS + 1):
            train_loss = self.train_epoch(train_loader)
            val_mae = self.validate(val_loader)

            print(
                f"Epoch {epoch}/{self.config.EPOCHS} | Train Loss: {train_loss} | Val MAE: {val_mae}"
            )

            # Save best model
            if val_mae < best_mae:
                best_mae = val_mae
                print(f"New best model found! Saving to {self.config.MODEL_PATH}")
                torch.save(self.model.state_dict(), self.config.MODEL_PATH)

        print(f"Training complete. Best Validation MAE: {best_mae}")

    def predict(self, load_cached_data=True):
        """
        Generates predictions for the test set using the best saved model.
        """
        print("Starting prediction phase...")

        if not os.path.exists(self.config.MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {self.config.MODEL_PATH}. Run fit() first."
            )

        # Load best model weights
        self.model.load_state_dict(
            torch.load(self.config.MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        # Get test data
        test_loader, test_ids = get_test_loader(self.config, load_cached_data)

        preds_list = []

        print("Running inference...")
        with torch.no_grad():
            for X, u_out in test_loader:
                X = X.to(self.device)
                preds = self.model(X)
                preds_list.append(preds.squeeze(-1).cpu().numpy())

        # Flatten predictions and IDs to match submission format
        # preds_list structure: List of (Batch, 80) -> Concat -> (N_breaths, 80) -> Flatten -> (N_steps,)
        flat_preds = np.concatenate(preds_list).flatten()
        flat_ids = test_ids.flatten()

        # Create DataFrame
        submission = pd.DataFrame({"id": flat_ids, "pressure": flat_preds})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)

        # Save submission
        print(f"Saving submission file to {self.config.SUBMISSION_PATH}")
        submission.to_csv(self.config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")
