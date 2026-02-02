import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, WeightedL1Loss, compute_metric
from library.dataset import load_data
from library.model import FPBC_BiLSTM


class Trainer:
    """
    Manages the training, validation, and submission generation lifecycle
    for the Ventilator Pressure Prediction task.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Initialize Model
        self.model = FPBC_BiLSTM().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler (Stretched Horizon)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Initialize Loss
        self.criterion = WeightedL1Loss()

        # Best metric tracking
        self.best_val_mae = float("inf")

    def train_epoch(self, train_loader):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for X, u_out, y in train_loader:
            X = X.to(self.device)
            u_out = u_out.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(X, u_out)

            # Calculate loss
            loss = self.criterion(preds, y, u_out)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.CLIP_GRAD_NORM
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self, val_loader):
        """Runs validation and computes MAE."""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_batches = 0

        with torch.no_grad():
            for X, u_out, y in val_loader:
                X = X.to(self.device)
                u_out = u_out.to(self.device)
                y = y.to(self.device)

                preds = self.model(X, u_out)

                # Loss for monitoring
                loss = self.criterion(preds, y, u_out)
                total_loss += loss.item()

                # Metric (MAE on inspiratory phase)
                mae = compute_metric(preds, y, u_out)
                total_mae += mae
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        return avg_loss, avg_mae

    def fit(self):
        """Main training loop."""
        print("Starting training process...")

        # Load Data
        train_dataset, val_dataset, test_dataset = load_data(load_cached_data=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Training Loop
        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate(val_loader)

            # Step Scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MAE: {val_mae}"
            )

            # Save Best Model
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"New best model saved with MAE: {self.best_val_mae}")

        print("Training complete.")

        # Generate Submission
        self.generate_submission(test_dataset)

    def generate_submission(self, test_dataset):
        """Generates predictions for the test set and saves submission.csv."""
        print("Generating submission...")

        # Load Best Model
        if not os.path.exists(Config.BEST_MODEL_PATH):
            raise FileNotFoundError("Best model checkpoint not found.")

        self.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        all_preds = []

        with torch.no_grad():
            for X, u_out in test_loader:
                X = X.to(self.device)
                u_out = u_out.to(self.device)

                # Predict
                preds = self.model(X, u_out)

                # Flatten: (Batch, 80) -> (Batch * 80)
                preds_flat = preds.view(-1).cpu().numpy()
                all_preds.append(preds_flat)

        # Concatenate all predictions
        final_predictions = np.concatenate(all_preds)

        # Align with IDs
        # We load the cached test dataframe because it preserves the exact sort order used during dataset creation
        # (sorted by breath_id, then time_step)
        if not os.path.exists(Config.TEST_CACHE_PATH):
            # Fallback if cache deleted, though unlikely given flow
            raise FileNotFoundError(
                f"Test cache not found at {Config.TEST_CACHE_PATH}. Cannot align IDs."
            )

        df_test = pd.read_parquet(Config.TEST_CACHE_PATH)

        # Sanity check on lengths
        if len(df_test) != len(final_predictions):
            raise ValueError(
                f"Mismatch: Test DF has {len(df_test)} rows, Predictions have {len(final_predictions)}"
            )

        # Create submission dataframe
        submission = pd.DataFrame(
            {
                Config.ID_COL: df_test[Config.ID_COL],
                Config.TARGET_COL: final_predictions,
            }
        )

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    trainer = Trainer()
    trainer.fit()


if __name__ == "__main__":
    main()
