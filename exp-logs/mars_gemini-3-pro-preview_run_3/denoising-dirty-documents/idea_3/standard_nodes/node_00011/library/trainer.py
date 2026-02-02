import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.model import DRDN
from library.dataset import get_dataloaders


class Trainer:
    """
    Manages the training and validation lifecycle of the DRDN model.
    """

    def __init__(self):
        """
        Initialize the Trainer with model, optimizer, loss function, and configuration.
        """
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = DRDN().to(self.device)

        # Hyperparameters
        self.epochs = Config.NUM_EPOCHS
        self.patience = Config.EARLY_STOPPING_PATIENCE
        self.learning_rate = Config.LEARNING_RATE
        self.weight_decay = Config.WEIGHT_DECAY
        self.save_path = Config.MODEL_SAVE_PATH

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

        # Loss Function: MSE Loss for noise prediction
        self.criterion = nn.MSELoss()

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.

        Args:
            train_loader (DataLoader): The training data loader.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # The model predicts the noise residual.
            # Ground Truth Noise = Noisy Input - Clean Target
            noise_target = inputs - targets

            self.optimizer.zero_grad()

            # Forward pass
            noise_pred = self.model(inputs)

            # Calculate loss
            loss = self.criterion(noise_pred, noise_target)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        avg_loss = running_loss / total_samples if total_samples > 0 else 0.0
        return avg_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set using RMSE.

        Args:
            val_loader (DataLoader): The validation data loader.

        Returns:
            float: The calculated RMSE over the validation set.
        """
        self.model.eval()
        mse_accum = 0.0
        total_pixels = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                # Predict noise
                noise_pred = self.model(inputs)

                # Reconstruct clean image: Input - Predicted Noise
                clean_pred = inputs - noise_pred

                # Clamp values to valid range [0, 1]
                clean_pred = torch.clamp(clean_pred, 0.0, 1.0)

                # Calculate Squared Error for RMSE calculation
                # Metric is RMSE between cleaned pixel intensities and actual grayscale intensities
                # Scale to [0, 255] for metric calculation
                clean_pred_scaled = clean_pred * 255.0
                targets_scaled = targets * 255.0
                se = (clean_pred_scaled - targets_scaled) ** 2
                mse_accum += torch.sum(se).item()
                total_pixels += torch.numel(targets)

        # Calculate RMSE
        rmse = (
            float(np.sqrt(mse_accum / total_pixels))
            if total_pixels > 0
            else float("inf")
        )
        return rmse

    def fit(self, load_cached_data=True, debug_limit=None):
        """
        Executes the full training pipeline with Early Stopping.

        Args:
            load_cached_data (bool): Whether to load pre-processed data from cache.
            debug_limit (int, optional): Limit the number of samples for debugging.
        """
        set_seed(Config.SEED)

        # Load Data
        print("Initializing DataLoaders...")
        train_loader, val_loader = get_dataloaders(
            load_cached_data=load_cached_data, debug_limit=debug_limit
        )

        best_val_rmse = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device} for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_rmse = self.validate(val_loader)

            elapsed = time.time() - start_time

            # Print metrics (Full precision)
            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {train_loss} - "
                f"Val RMSE: {val_rmse} - "
                f"Time: {elapsed:.2f}s"
            )

            # Update Scheduler
            self.scheduler.step(val_rmse)

            # Early Stopping and Checkpointing
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_rmse, self.save_path
                )
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation RMSE: {best_val_rmse}")
