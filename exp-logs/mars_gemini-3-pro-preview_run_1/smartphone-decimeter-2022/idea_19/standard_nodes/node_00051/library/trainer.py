import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from library.config import Config
from library.model import SkyResUNet


class Trainer:
    """
    Trainer class for the Cyclic Spatio-Temporal 2D ResUNet.
    Handles training loop, validation, deep supervision loss computation,
    and checkpointing.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.model = SkyResUNet().to(self.device)

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

        # Loss function (Mean Absolute Error)
        self.criterion = nn.L1Loss(reduction="none")

        self.best_val_loss = float("inf")
        self.model_path = os.path.join(Config.MODEL_DIR, "best_model.pth")

    def _downsample(self, tensor, scale_factor):
        """
        Downsamples a tensor along the temporal dimension (dim 1) using Average Pooling.

        Args:
            tensor: (Batch, Time, Channels) or (Batch, Time)
            scale_factor: Integer factor to downsample by.

        Returns:
            Downsampled tensor.
        """
        if scale_factor == 1:
            return tensor

        # Permute to (Batch, Channels, Time) for pool1d
        if tensor.dim() == 3:
            x = tensor.permute(0, 2, 1)
            x = F.avg_pool1d(x, kernel_size=scale_factor, stride=scale_factor)
            return x.permute(0, 2, 1)
        elif tensor.dim() == 2:
            # Add channel dim, pool, remove channel dim
            x = tensor.unsqueeze(1)
            x = F.avg_pool1d(x, kernel_size=scale_factor, stride=scale_factor)
            return x.squeeze(1)
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")

    def _compute_loss(self, outputs, targets, mask):
        """
        Computes the weighted sum of losses from main and auxiliary heads.

        Args:
            outputs: Dict containing 'main', 'aux2', 'aux3', 'aux4' (if training)
                     or Tensor (if eval).
            targets: Ground truth (Batch, Time, 2).
            mask: Validity mask (Batch, Time).

        Returns:
            Total weighted loss.
        """
        # Handle Eval Mode (Single output tensor)
        if not isinstance(outputs, dict):
            loss = self.criterion(outputs, targets)
            # Apply mask: loss is (B, T, 2), mask is (B, T)
            # Expand mask to (B, T, 2)
            mask_expanded = mask.unsqueeze(-1).expand_as(loss)
            masked_loss = (loss * mask_expanded).sum()

            # Normalize by number of valid elements
            valid_elements = mask_expanded.sum()
            if valid_elements > 0:
                return masked_loss / valid_elements
            else:
                return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Handle Training Mode (Deep Supervision)
        total_loss = 0.0

        # Define scales and weights for deep supervision
        # Keys match the model output keys
        # Scales: 1 (Main), 2 (Aux2), 4 (Aux3), 8 (Aux4)
        heads = [
            ("main", 1, 1.0),
            ("aux2", 2, 0.5),
            ("aux3", 4, 0.25),
            ("aux4", 8, 0.125),
        ]

        for name, scale, weight in heads:
            if name not in outputs:
                continue

            pred = outputs[name]

            # Downsample target and mask to match prediction resolution
            target_ds = self._downsample(targets, scale)
            mask_ds = self._downsample(mask, scale)

            # Ensure shapes match (handle potential odd-dimension pooling mismatches)
            if pred.shape[1] != target_ds.shape[1]:
                # Truncate to minimum length
                min_len = min(pred.shape[1], target_ds.shape[1])
                pred = pred[:, :min_len, :]
                target_ds = target_ds[:, :min_len, :]
                mask_ds = mask_ds[:, :min_len]

            loss = self.criterion(pred, target_ds)

            # Apply mask
            mask_expanded = mask_ds.unsqueeze(-1).expand_as(loss)
            masked_loss = (loss * mask_expanded).sum()

            valid_elements = mask_expanded.sum()
            if valid_elements > 0:
                term_loss = masked_loss / valid_elements
            else:
                term_loss = torch.tensor(0.0, device=self.device)

            total_loss += weight * term_loss

        return total_loss

    def train_epoch(self, dataloader):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features)
            loss = self._compute_loss(outputs, targets, mask)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            self.optimizer.step()

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def validate_epoch(self, dataloader):
        """Runs validation."""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model(features)
                loss = self._compute_loss(outputs, targets, mask)

                running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def fit(self, train_loader, val_loader):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate_epoch(val_loader)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

            # Checkpoint
            if val_loss < self.best_val_loss:
                print(
                    f"Validation loss improved from {self.best_val_loss:.6f} to {val_loss:.6f}. Saving model..."
                )
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")

    def load_best_model(self):
        """Loads the best saved model weights."""
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            print(f"Loaded best model from {self.model_path}")
        else:
            print("No saved model found.")

    def predict(self, dataloader):
        """
        Generates predictions for a dataloader.
        Returns concatenated predictions, WLS positions, and timestamps.
        """
        self.model.eval()
        all_preds = []
        all_wls = []
        all_timestamps = []
        all_masks = []

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                mask = batch["mask"]
                wls = batch["wls_pos"]
                ts = batch["timestamps"]

                # Forward pass
                # Output shape: (Batch, Time, 2) -> (dEast, dNorth)
                residuals = self.model(features).cpu().numpy()

                all_preds.append(residuals)
                all_wls.append(wls.numpy())
                all_timestamps.append(ts.numpy())
                all_masks.append(mask.numpy())

        return (
            np.concatenate(all_preds, axis=0),
            np.concatenate(all_wls, axis=0),
            np.concatenate(all_timestamps, axis=0),
            np.concatenate(all_masks, axis=0),
        )
