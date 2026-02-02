import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import time

from library.config import Config
from library.dataset import prepare_datasets
from library.model import WPABiLSTM
from library.loss import WeightedL1Loss


class Trainer:
    def __init__(self):
        # Initialize configuration and directories
        Config.setup()
        self.device = torch.device(Config.DEVICE)

        # Load datasets
        print("Initializing Trainer and loading datasets...")
        self.train_dataset, self.val_dataset, self.test_dataset = prepare_datasets(
            load_cached_data=True
        )

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Initialize Model
        self.model = WPABiLSTM().to(self.device)

        # Initialize Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Initialize Loss
        self.criterion = WeightedL1Loss().to(self.device)

        # Best metric tracking
        self.best_val_mae = float("inf")

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(inputs)

            # Calculate loss (requires inputs for u_out weighting)
            loss = self.criterion(preds, targets, inputs)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0.0
        total_insp_mae = 0.0
        total_insp_count = 0

        # Index of u_out in the input features
        # Continuous features come first, then binary features
        u_out_idx = len(Config.CONTINUOUS_FEATURES) + Config.BINARY_FEATURES.index(
            "u_out"
        )

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                preds = self.model(inputs)

                # Weighted Loss
                loss = self.criterion(preds, targets, inputs)
                total_loss += loss.item() * inputs.size(0)

                # Competition Metric: MAE on inspiratory phase (u_out == 0)
                u_out = inputs[:, :, u_out_idx]
                mask = u_out == 0

                abs_err = torch.abs(preds - targets)
                insp_err = abs_err[mask]

                total_insp_mae += insp_err.sum().item()
                total_insp_count += insp_err.numel()

        avg_loss = total_loss / len(self.val_dataset)
        avg_insp_mae = (
            total_insp_mae / total_insp_count if total_insp_count > 0 else 0.0
        )

        return avg_loss, avg_insp_mae

    def fit(self):
        print(f"Starting training for {Config.EPOCHS} epochs on {self.device}...")

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_insp_mae = self.validate_epoch()

            # Step Scheduler
            self.scheduler.step()

            epoch_time = time.time() - start_time

            # Print metrics
            print(
                f"Epoch {epoch}/{Config.EPOCHS} | Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"Val Insp MAE: {val_insp_mae}"
            )

            # Save Best Model (based on Inspiratory MAE)
            if val_insp_mae < self.best_val_mae:
                self.best_val_mae = val_insp_mae
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"New best model saved! MAE: {self.best_val_mae}")

    def predict(self):
        """
        Generates predictions for the test set using the best model and saves to submission.csv.
        """
        print("Starting prediction on test set...")

        # Load best model
        if not os.path.exists(Config.BEST_MODEL_PATH):
            print("No best model found. Skipping prediction.")
            return

        self.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        all_preds = []

        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(self.device)
                preds = self.model(inputs)
                # Flatten predictions: (batch, seq) -> (batch * seq)
                all_preds.append(preds.cpu().numpy().flatten())

        # Concatenate all predictions
        final_preds = np.concatenate(all_preds)

        # Load Test Metadata to map to IDs
        print("Loading test metadata...")
        test_meta = pd.read_csv(Config.TEST_META)

        # Ensure lengths match
        if len(final_preds) != len(test_meta):
            print(
                f"Warning: Prediction length {len(final_preds)} does not match metadata length {len(test_meta)}."
            )

        # Create submission DataFrame
        submission = pd.DataFrame({"id": test_meta["id"], "pressure": final_preds})

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")


def main():
    trainer = Trainer()
    trainer.fit()
    trainer.predict()
