import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from library.config import Config
from library.utils import set_seed, get_device
from library.data_loader import get_data_loaders
from library.model import PhysicsInjectedNet


class MaskedL1Loss(nn.Module):
    """
    Computes the Mean Absolute Error (L1 Loss) strictly for the inspiratory phase.
    The expiratory phase (u_out == 1) is masked out and does not contribute to the gradient.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets, u_out):
        """
        Args:
            preds (torch.Tensor): Model predictions of shape (Batch, Seq).
            targets (torch.Tensor): Ground truth pressure of shape (Batch, Seq).
            u_out (torch.Tensor): Control input 'u_out' of shape (Batch, Seq).
                                  0 indicates inspiration, 1 indicates expiration.
        """
        # Create mask: 1 for inspiratory (u_out=0), 0 for expiratory (u_out=1)
        mask = 1 - u_out

        # Calculate element-wise absolute error
        mae = torch.abs(preds - targets)

        # Apply mask
        masked_mae = mae * mask

        # Compute mean only over the valid (inspiratory) time steps
        # Add a small epsilon to denominator to prevent division by zero
        loss = masked_mae.sum() / (mask.sum() + 1e-8)

        return loss


class Trainer:
    """
    Manages the training, validation, and checkpointing process.
    """

    def __init__(self, train_loader, val_loader):
        self.device = get_device()
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize the Physics-Injected Model
        self.model = PhysicsInjectedNet().to(self.device)

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: OneCycleLR
        # This scheduler anneals the learning rate according to the 1cycle policy,
        # which is beneficial for convergence in deep residual networks.
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            epochs=Config.EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=Config.PCT_START,
            div_factor=Config.DIV_FACTOR,
            final_div_factor=Config.FINAL_DIV_FACTOR,
        )

        # Loss Function
        self.criterion = MaskedL1Loss()

        # Determine the index of 'u_out' in the feature tensor for masking
        try:
            self.u_out_idx = Config.FEATURE_COLS.index("u_out")
        except ValueError:
            raise ValueError(
                "Feature 'u_out' is required in Config.FEATURE_COLS for MaskedL1Loss."
            )

    def train_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)

            # Extract u_out for the loss mask (Batch, Seq)
            # X shape is (Batch, Seq, Features)
            u_out = X[:, :, self.u_out_idx]

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(X)

            # Compute masked loss
            loss = self.criterion(preds, y, u_out)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self):
        """Evaluates the model on the validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for X, y in self.val_loader:
                X, y = X.to(self.device), y.to(self.device)

                u_out = X[:, :, self.u_out_idx]

                preds = self.model(X)
                loss = self.criterion(preds, y, u_out)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def fit(self, patience=10):
        """
        Runs the full training loop with early stopping and model checkpointing.

        Args:
            patience (int): Number of epochs to wait for improvement before early stopping.
        """
        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs.")
        print(f"Batch Size: {Config.BATCH_SIZE}, Learning Rate: {Config.LEARNING_RATE}")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Print full precision metrics
            print(
                f"Epoch {epoch}/{Config.EPOCHS} - Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(
                    f"New best model saved to {Config.MODEL_PATH} with Val Loss: {best_val_loss}"
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping triggered. No improvement for {patience} epochs."
                    )
                    break

        print(f"Training complete. Best Validation Loss: {best_val_loss}")


def train_model():
    """
    Main entry point to setup environment, load data, and start training.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Load Datasets
    # load_cached=True ensures we use the pre-processed .npy files if available
    train_loader, val_loader, _ = get_data_loaders(
        batch_size=Config.BATCH_SIZE, load_cached=True
    )

    # Initialize Trainer
    trainer = Trainer(train_loader, val_loader)

    # Execute Training
    # We use a patience of 10, which is reasonable for a 35-epoch OneCycle schedule
    trainer.fit(patience=10)
