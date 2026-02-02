import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.model import TransUNet1D
from library.data_loader import get_dataloaders
from library.utils import set_seed, enu_to_geodetic


class Trainer:
    """
    Manages the training and validation of the TransUNet1D model.
    """

    def __init__(self, model, device, optimizer, scheduler=None, criterion=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Use HuberLoss as specified in the design
        self.criterion = (
            criterion if criterion else nn.HuberLoss(delta=Config.HUBER_DELTA)
        )
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, dataloader):
        """
        Runs one training epoch.
        """
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            features, targets = batch
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            # Model input: (B, C, L)
            outputs = self.model(features)

            # Targets from loader are (B, L, C). Align to (B, C, L) for loss calculation
            targets = targets.permute(0, 2, 1)

            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def validate_epoch(self, dataloader):
        """
        Runs validation and calculates metrics.
        """
        self.model.eval()
        running_loss = 0.0
        total_dist_error = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                features, targets = batch
                features = features.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(features)

                # Loss calculation (B, C, L)
                targets_permuted = targets.permute(0, 2, 1)
                loss = self.criterion(outputs, targets_permuted)
                running_loss += loss.item() * features.size(0)

                # Metric calculation: Mean Euclidean Distance in Meters
                # Outputs are residuals (DeltaEast, DeltaNorth)
                # Shape for calculation: (B, L, 2)
                outputs_np = outputs.permute(0, 2, 1).cpu().numpy()
                targets_np = targets.cpu().numpy()

                # Calculate Euclidean distance between predicted residual and target residual
                # This is equivalent to distance between Predicted Position and Ground Truth
                diff = outputs_np - targets_np
                dist_error = np.sqrt(np.sum(diff**2, axis=2))  # Shape: (B, L)

                # Average error per sequence, then sum up for the epoch
                total_dist_error += np.sum(np.mean(dist_error, axis=1))
                total_samples += features.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_metric = total_dist_error / total_samples if total_samples > 0 else 0.0
        return epoch_loss, epoch_metric

    def fit(self, train_loader, val_loader, num_epochs, checkpoint_path):
        """
        Main training loop with early stopping.
        """
        print(f"Starting training on {self.device}...")

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_metric = self.validate_epoch(val_loader)

            if self.scheduler:
                self.scheduler.step(val_loss)

            duration = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Dist Error: {val_metric:.6f} m | "
                f"Time: {duration:.1f}s"
            )

            # Checkpoint and Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  -> Model saved to {checkpoint_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.PATIENCE:
                    print("  -> Early stopping triggered.")
                    break


def run_training():
    """
    Sets up the pipeline and runs training.
    """
    set_seed(Config.SEED)

    # Load data
    print("Loading data...")
    # load_cached_data=True allows skipping preprocessing if files exist
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Initialize model
    model = TransUNet1D().to(Config.DEVICE)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Trainer
    trainer = Trainer(model, Config.DEVICE, optimizer, scheduler)

    # Define checkpoint path
    checkpoint_path = os.path.join(Config.WORKING_DIR, "model_weights.pth")

    # Run training
    trainer.fit(train_loader, val_loader, Config.NUM_EPOCHS, checkpoint_path)

    return trainer, test_loader


def generate_submission(trainer, test_loader):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("\nGenerating submission...")

    # Load best weights
    checkpoint_path = os.path.join(Config.WORKING_DIR, "model_weights.pth")
    if os.path.exists(checkpoint_path):
        trainer.model.load_state_dict(
            torch.load(checkpoint_path, map_location=Config.DEVICE)
        )
        print("Loaded best model weights.")
    else:
        print("Warning: Checkpoint not found, using current model weights.")

    trainer.model.eval()

    # We need to map predictions back to trip IDs.
    # The test_loader iterates over drives in the order they appear in test_metadata.
    # We load test_metadata to get the drive_id and phone_name order.
    test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Get unique trips (drive+phone) preserving order
    unique_trips = test_meta_df[["drive_id", "phone_name"]].drop_duplicates()

    predictions = []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            # Batch size is 1 for test loader
            features = batch["features"].to(Config.DEVICE)  # (1, C, L)
            wls_lat = batch["wls_lat"].numpy()[0]  # (L,)
            wls_lon = batch["wls_lon"].numpy()[0]  # (L,)
            timestamps = batch["timestamps"].numpy()[0]  # (L,)

            # Identify trip
            if idx < len(unique_trips):
                meta_row = unique_trips.iloc[idx]
                drive_id = meta_row["drive_id"]
                phone_name = meta_row["phone_name"]
                trip_id = f"{drive_id}-{phone_name}"
            else:
                # Should not happen if loader and metadata are consistent
                continue

            # Model Inference
            outputs = trainer.model(features)
            outputs_np = outputs.cpu().numpy()[0]  # (2, L)

            pred_east = outputs_np[0, :]
            pred_north = outputs_np[1, :]

            # Convert predicted ENU residuals to Geodetic coordinates
            # New Lat/Lon = WLS Lat/Lon + Geodetic Offset(East, North)
            pred_lat, pred_lon = enu_to_geodetic(
                pred_east, pred_north, wls_lat, wls_lon
            )

            # Create DataFrame for this trip
            trip_df = pd.DataFrame(
                {
                    "tripId": trip_id,
                    "UnixTimeMillis": timestamps,
                    "LatitudeDegrees": pred_lat,
                    "LongitudeDegrees": pred_lon,
                }
            )

            predictions.append(trip_df)

    if predictions:
        all_preds = pd.concat(predictions, ignore_index=True)

        # Load sample submission to get the exact required rows and timestamps
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Data Loader rounds timestamps to nearest second (1000ms).
        # Sample submission might have raw timestamps.
        # Create a join key based on rounded timestamps.
        sample_sub["UnixTimeMillis_rounded"] = (
            np.round(sample_sub["UnixTimeMillis"] / 1000) * 1000
        )
        all_preds["UnixTimeMillis_rounded"] = (
            np.round(all_preds["UnixTimeMillis"] / 1000) * 1000
        )

        # Merge predictions into sample submission structure
        # Left join ensures we keep all rows from sample submission
        submission = pd.merge(
            sample_sub,
            all_preds[
                [
                    "tripId",
                    "UnixTimeMillis_rounded",
                    "LatitudeDegrees",
                    "LongitudeDegrees",
                ]
            ],
            on=["tripId", "UnixTimeMillis_rounded"],
            how="left",
            suffixes=("_old", ""),
        )

        # Select final columns
        submission = submission[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ]

        # Fill any missing predictions (if merge failed) with 0 or forward fill if appropriate
        # Here we just fill with 0 to avoid NaNs in output, though in a real scenario we might fallback to WLS
        if submission.isnull().any().any():
            print(
                f"Warning: {submission.isnull().sum().sum()} missing predictions found. Filling with 0."
            )
            submission = submission.fillna(0)

        # Save submission
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
        print(f"Submission shape: {submission.shape}")
    else:
        print("No predictions generated.")
