import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import seed_everything
from library.dataset import TaxiDataset
from library.model import SpatialResNet


class Trainer:
    """
    Trainer class for the SpatialResNet model.
    Handles training, validation, and submission generation.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = SpatialResNet(
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            num_res_blocks=Config.NUM_RES_BLOCKS,
            dropout_rate=Config.DROPOUT_RATE,
            grid_bins=Config.GRID_BINS,
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function: SmoothL1Loss (Huber Loss)
        # Behaves like L2 near zero (for stability) and L1 for large errors (for robustness against outliers)
        self.criterion = nn.SmoothL1Loss(beta=Config.HUBER_DELTA)

        # Track best performance for early stopping
        self.best_rmse = float("inf")

    def train_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = len(train_loader)

        start_time = time.time()

        for batch in train_loader:
            # Move data to device
            cont_feat = batch["continuous_features"].to(self.device)
            cat_idx = batch["spatial_indices"].to(self.device)
            targets = batch["target"].to(self.device).view(-1, 1)  # Ensure shape (B, 1)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(cont_feat, cat_idx)

            # Compute Loss
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Critical for stability with outliers)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.GRAD_CLIP_NORM
            )

            # Optimizer Step
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / num_batches
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch_idx+1}/{Config.EPOCHS} - Train Loss: {avg_loss:.6f} - Time: {elapsed:.2f}s"
        )
        return avg_loss

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set using RMSE.
        """
        self.model.eval()
        mse_sum = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                cont_feat = batch["continuous_features"].to(self.device)
                cat_idx = batch["spatial_indices"].to(self.device)
                targets = batch["target"].to(self.device).view(-1, 1)

                outputs = self.model(cont_feat, cat_idx)

                # Apply post-processing floor to align validation metric with actual usage
                preds = torch.clamp(outputs, min=Config.MIN_FARE_PREDICTION)

                # Accumulate Squared Errors
                mse_sum += torch.sum((preds - targets) ** 2).item()
                total_samples += targets.size(0)

        rmse = np.sqrt(mse_sum / total_samples)
        return rmse

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        seed_everything(Config.SEED)

        # Initialize Datasets and Loaders
        print("Loading Training Data...")
        train_dataset = TaxiDataset(split="train", load_cached_data=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print("Loading Validation Data...")
        val_dataset = TaxiDataset(split="val", load_cached_data=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Starting training on device: {self.device}")

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            _ = self.train_epoch(train_loader, epoch)
            val_rmse = self.validate(val_loader)

            print(f"Epoch {epoch+1} Validation RMSE: {val_rmse}")

            # Early Stopping Logic
            if val_rmse < self.best_rmse:
                print(
                    f"Validation RMSE improved from {self.best_rmse} to {val_rmse}. Saving model..."
                )
                self.best_rmse = val_rmse
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best RMSE: {self.best_rmse}")

    def generate_submission(self):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print("Generating submission...")

        # Load Test Data
        test_dataset = TaxiDataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Load Best Model
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print("No trained model found. Cannot generate submission.")
            return

        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        keys_list = []
        preds_list = []

        with torch.no_grad():
            for batch in test_loader:
                cont_feat = batch["continuous_features"].to(self.device)
                cat_idx = batch["spatial_indices"].to(self.device)
                keys = batch["key"]  # Tuple of strings

                outputs = self.model(cont_feat, cat_idx)

                # Apply Post-Processing (Floor)
                preds = torch.clamp(outputs, min=Config.MIN_FARE_PREDICTION)

                preds_list.extend(preds.cpu().numpy().flatten())
                keys_list.extend(keys)

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"key": keys_list, "fare_amount": preds_list})

        # Save to CSV
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
