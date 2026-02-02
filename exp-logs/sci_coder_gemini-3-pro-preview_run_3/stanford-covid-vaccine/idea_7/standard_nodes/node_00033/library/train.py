import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from library.config import Config
from library.utils import set_seed, mcrmse_metric
from library.dataset import get_dataloaders
from library.model import SpatiallyAugmentedBiGRU
from library.loss import WeightedMCRMSELoss


class Trainer:
    """
    Manages the training and validation process for the RNA degradation model.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX
        )

        # Loss function (Noise-Aware)
        self.criterion = WeightedMCRMSELoss()

        # Early Stopping parameters
        self.patience = 5
        self.best_metric = float("inf")
        self.counter = 0

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (X, y, w) in enumerate(self.train_loader):
            X = X.to(self.device)
            y = y.to(self.device)
            w = w.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(X)

            # Calculate Weighted Loss
            loss = self.criterion(preds, y, w)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            # Optimizer step
            self.optimizer.step()

            running_loss += loss.item()

        # Step scheduler at the end of epoch
        self.scheduler.step()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Runs validation and calculates the MCRMSE metric.
        """
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y, _ in self.val_loader:
                X = X.to(self.device)

                # Forward pass
                preds = self.model(X)

                # Collect predictions and targets for metric calculation
                # We only need the scored part for the metric, but the metric utility
                # handles slicing if we pass full arrays, or we can slice here.
                # The metric utility expects (N, seq_scored, 5) or similar.
                # Model output is (N, 107, 5). Targets are (N, 68, 5).

                # Slice predictions to match target length (68)
                preds_sliced = preds[:, : Config.SEQ_SCORED, :]

                all_preds.append(preds_sliced.cpu().numpy())
                all_targets.append(y.numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate Metric
        metric = mcrmse_metric(all_targets, all_preds)
        return metric

    def fit(self, epochs=Config.EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metric = self.validate()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE: {val_metric} | "
                f"Time: {elapsed:.2f}s"
            )

            # Early Stopping and Model Saving
            if val_metric < self.best_metric:
                self.best_metric = val_metric
                self.counter = 0
                print(f"New best model found! Saving to {Config.MODEL_SAVE_PATH}")
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Val MCRMSE: {self.best_metric}")


def train_model():
    """
    Sets up the environment and executes the training pipeline.
    """
    # 1. Setup System
    Config.setup_system()
    set_seed(Config.SEED)

    # 2. Load Data
    print("Loading data...")
    loaders = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # 3. Initialize Model
    print("Initializing model...")
    device = torch.device(Config.DEVICE)
    model = SpatiallyAugmentedBiGRU().to(device)

    # 4. Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, device)

    # 5. Run Training
    trainer.fit(epochs=Config.EPOCHS)


# Note: The if __name__ == "__main__": block is omitted as per instructions.
