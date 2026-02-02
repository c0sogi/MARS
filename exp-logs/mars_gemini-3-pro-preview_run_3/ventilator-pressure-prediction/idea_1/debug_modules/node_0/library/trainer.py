import torch
import torch.nn as nn
import os
from library.config import Config
from library.utils import AverageMeter


class Trainer:
    """
    Trainer class to handle the training and validation loops with Masked L1 Loss.
    """

    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        # Use reduction='none' to apply mask later
        self.criterion = nn.L1Loss(reduction="none")
        # Identify index of u_out for masking
        try:
            self.u_out_idx = Config.FEATURE_COLS.index("u_out")
        except ValueError:
            raise ValueError("u_out not found in Config.FEATURE_COLS")

    def get_inspiratory_mask(self, x):
        """
        Generates a boolean mask for the inspiratory phase (u_out == 0).

        Args:
            x (torch.Tensor): Input features of shape (batch, seq_len, features).

        Returns:
            torch.Tensor: Boolean mask of shape (batch, seq_len).
        """
        # Extract u_out feature
        u_out = x[:, :, self.u_out_idx]

        # Since features are normalized (mean ~0.62, std ~0.48),
        # u_out=0 becomes negative (~-1.2) and u_out=1 becomes positive (~0.8).
        # We use 0.0 as a safe threshold to distinguish between the two binary states in normalized space.
        return u_out < 0.0

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        meter = AverageMeter()

        for x, y in train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(x)

            # Generate mask for inspiratory phase
            mask = self.get_inspiratory_mask(x)

            # Compute element-wise loss
            raw_loss = self.criterion(preds, y)

            # Apply mask: select only inspiratory time steps
            masked_loss = raw_loss[mask]

            # Check if we have valid items (should always be true for this dataset)
            if masked_loss.numel() > 0:
                loss = masked_loss.mean()

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Update metrics
                # We track the mean loss weighted by the number of valid items
                meter.update(loss.item(), masked_loss.numel())

        return meter.avg

    def validate_epoch(self, val_loader):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        meter = AverageMeter()

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                preds = self.model(x)
                mask = self.get_inspiratory_mask(x)

                raw_loss = self.criterion(preds, y)
                masked_loss = raw_loss[mask]

                if masked_loss.numel() > 0:
                    meter.update(masked_loss.mean().item(), masked_loss.numel())

        return meter.avg

    def fit(self, train_loader, val_loader):
        """
        Full training loop with Early Stopping.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            # Print full precision metrics
            print(
                f"Epoch {epoch + 1}/{Config.EPOCHS} - "
                f"Train MAE: {train_loss} - "
                f"Val MAE: {val_loss}"
            )

            # Early Stopping and Model Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best Val MAE: {best_val_loss}"
                    )
                    break

        print(f"Training complete. Best Val MAE: {best_val_loss}")
