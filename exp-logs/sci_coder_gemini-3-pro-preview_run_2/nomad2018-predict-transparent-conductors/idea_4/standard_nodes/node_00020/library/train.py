import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import Standardizer, compute_rmsle
from library.model import CrystalGraphConvNet
from library.data import get_dataloaders


class Trainer:
    """
    Manages the training lifecycle of the Crystal Graph ConvNet.
    """

    def __init__(self, device=None):
        self.device = device if device else Config.DEVICE
        self.model = CrystalGraphConvNet().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
        )
        self.criterion = nn.MSELoss()
        self.standardizer = Standardizer(device=self.device)
        self.best_val_loss = float("inf")
        self.checkpoint_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    def fit_standardizer(self, train_loader):
        """
        Fits the target standardizer using data from the training loader.
        """
        all_targets = []
        for batch in train_loader:
            all_targets.append(batch.y)

        all_targets = torch.cat(all_targets, dim=0)
        self.standardizer.fit(all_targets)
        print(
            f"Standardizer fitted. Mean: {self.standardizer.mean.cpu().numpy()}, Std: {self.standardizer.std.cpu().numpy()}"
        )

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(self.device)

            # Standardize targets
            targets_norm = self.standardizer.transform(batch.y)

            self.optimizer.zero_grad()
            preds_norm = self.model(batch)

            loss = self.criterion(preds_norm, targets_norm)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def evaluate(self, loader):
        """
        Evaluates the model on a given loader.
        Returns average MSE loss (standardized) and RMSLE (original scale).
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                # Forward pass
                preds_norm = self.model(batch)

                # Calculate standardized MSE loss (used for scheduling/early stopping)
                targets_norm = self.standardizer.transform(batch.y)
                loss = self.criterion(preds_norm, targets_norm)
                total_loss += loss.item()

                # Inverse transform for RMSLE calculation
                preds = self.standardizer.inverse_transform(preds_norm)

                all_preds.append(preds.cpu())
                all_targets.append(batch.y.cpu())
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Concatenate for metric calculation
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        rmsle = compute_rmsle(all_targets, all_preds)

        return avg_loss, rmsle

    def train_loop(self, train_loader, val_loader, epochs=Config.NUM_EPOCHS):
        """
        Main training loop with early stopping.
        """
        # Ensure standardizer is fitted
        if self.standardizer.mean is None:
            print("Fitting standardizer on training data...")
            self.fit_standardizer(train_loader)

        patience_counter = 0

        print("Starting training...")
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_rmsle = self.evaluate(val_loader)

            # Scheduler step
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]

            duration = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss (MSE): {train_loss:.6f} | "
                f"Val Loss (MSE): {val_loss:.6f} | "
                f"Val RMSLE: {val_rmsle:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {duration:.2f}s"
            )

            # Checkpoint and Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "standardizer_mean": self.standardizer.mean,
                        "standardizer_std": self.standardizer.std,
                        "val_loss": val_loss,
                        "val_rmsle": val_rmsle,
                    },
                    self.checkpoint_path,
                )
                print(f"  -> New best model saved to {self.checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

    def load_best_model(self):
        """
        Loads the best model checkpoint.
        """
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load standardizer stats to ensure consistency during inference
        self.standardizer.mean = checkpoint["standardizer_mean"].to(self.device)
        self.standardizer.std = checkpoint["standardizer_std"].to(self.device)

        print(
            f"Loaded model from epoch {checkpoint['epoch']} with Val RMSLE: {checkpoint['val_rmsle']:.6f}"
        )

    def predict(self, loader):
        """
        Generates predictions for a dataset.
        Returns: (ids, predictions)
        """
        self.model.eval()
        all_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)

                # Forward pass
                preds_norm = self.model(batch)

                # Inverse transform
                preds = self.standardizer.inverse_transform(preds_norm)

                all_ids.append(batch.id.cpu())
                all_preds.append(preds.cpu())

        return torch.cat(all_ids, dim=0).numpy(), torch.cat(all_preds, dim=0).numpy()


def train_model(load_cached_data=True, epochs=Config.NUM_EPOCHS):
    """
    Orchestrates the training pipeline.
    """
    # 1. Get Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # 2. Initialize Trainer
    trainer = Trainer()

    # 3. Train
    trainer.train_loop(train_loader, val_loader, epochs=epochs)

    # 4. Load best model for verification
    trainer.load_best_model()

    return trainer, test_loader


def generate_submission(
    trainer, test_loader, output_path="./submission/submission.csv"
):
    """
    Generates submission file using the trained model.
    """
    print("Generating predictions for test set...")
    ids, preds = trainer.predict(test_loader)

    # preds shape: (N, 2) -> [formation_energy, bandgap_energy]
    df = pd.DataFrame(
        {
            "id": ids.flatten(),
            "formation_energy_ev_natom": preds[:, 0],
            "bandgap_energy_ev": preds[:, 1],
        }
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sort by ID to match sample submission format usually
    df = df.sort_values("id")

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
