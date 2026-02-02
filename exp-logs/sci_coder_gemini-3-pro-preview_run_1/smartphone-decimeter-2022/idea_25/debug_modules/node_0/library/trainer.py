import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Dict

from library.config import Config
from library.model import StratifiedAttentionResUNet
from library.loss import DecimatedDeepSupervisionLoss
from library.dataset import GNSSSequenceDataset, GNSSScaler
from library.utils import enu_to_wgs84


class Trainer:
    def __init__(self, load_cached_data: bool = True):
        """
        Initializes the Trainer with model, optimizer, loss, and data loaders.

        Args:
            load_cached_data (bool): Whether to load pre-processed data from parquet cache.
        """
        self.device = Config.DEVICE
        self.load_cached_data = load_cached_data

        # 1. Prepare Data
        print("Initializing Datasets...")
        self.train_dataset = GNSSSequenceDataset(
            split="train", load_cached_data=self.load_cached_data
        )

        # Use the scaler fitted on training data for validation/test
        self.val_dataset = GNSSSequenceDataset(
            split="val",
            scaler=self.train_dataset.scaler,
            load_cached_data=self.load_cached_data,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # 2. Initialize Model
        print(f"Initializing Model: {Config.MODEL_NAME}")
        self.model = StratifiedAttentionResUNet(
            in_channels=Config.IN_CHANNELS,
            out_channels=Config.OUT_CHANNELS,
            filters=Config.ENCODER_FILTERS,
            deep_supervision=Config.DEEP_SUPERVISION,
        ).to(self.device)

        # 3. Setup Optimization
        self.criterion = DecimatedDeepSupervisionLoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.COSINE_T_MAX, eta_min=Config.COSINE_ETA_MIN
        )

        self.best_score = float("inf")
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_epoch(self, epoch_idx: int) -> float:
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, (X, targets, metadata) in enumerate(self.train_loader):
            # Move inputs to device
            X = X.to(self.device)

            # Move targets (dictionary of scales) to device
            if isinstance(targets, dict):
                targets = {k: v.to(self.device) for k, v in targets.items()}
            else:
                targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            preds = self.model(X)

            # Compute loss
            loss = self.criterion(preds, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.GRADIENT_CLIPPING
            )

            self.optimizer.step()

            # Track metrics
            # Loss is mean over batch, multiply by batch size to accumulate
            batch_size = X.size(0)
            running_loss += loss.item() * batch_size
            count += batch_size

        epoch_loss = running_loss / count
        return epoch_loss

    def validate(self) -> float:
        """
        Runs validation and computes the competition metric.
        Metric: Mean of 50th and 95th percentile distance errors.
        """
        self.model.eval()
        all_errors = []

        with torch.no_grad():
            for X, targets, metadata in self.val_loader:
                X = X.to(self.device)

                # Forward pass
                preds = self.model(X)

                # Get high-res prediction (scale 0)
                # Shape: (B, 2, L) -> (Delta East, Delta North)
                pred_high_res = preds[0].cpu().numpy()

                # Get ground truth high-res (scale 0)
                if isinstance(targets, dict):
                    gt_high_res = targets["scale_0"].numpy()
                else:
                    gt_high_res = targets.numpy()

                # Get padding mask
                pad_mask = metadata["pad_mask"].numpy()  # (B, L)

                # Compute Euclidean distance error for valid points
                # pred: (B, 2, L), gt: (B, 2, L)
                # diff: (B, 2, L)
                diff = pred_high_res - gt_high_res

                # dist: (B, L)
                dist = np.sqrt(np.sum(diff**2, axis=1))

                # Flatten and filter by mask
                valid_dist = dist[pad_mask.astype(bool)]
                all_errors.append(valid_dist)

        # Concatenate all errors
        if len(all_errors) > 0:
            all_errors = np.concatenate(all_errors)

            # Compute Percentiles
            p50 = np.percentile(all_errors, 50)
            p95 = np.percentile(all_errors, 95)
            score = (p50 + p95) / 2.0
        else:
            score = float("inf")

        return score

    def run(self):
        """Main training loop."""
        print(f"Starting training for {Config.EPOCHS} epochs...")

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_score = self.validate()

            # Scheduler Step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Time: {elapsed:.1f}s | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Score (Mean 50/95): {val_score:.6f}"
            )

            # Checkpointing & Early Stopping
            if val_score < self.best_score:
                print(
                    f"  >>> New Best Score! ({self.best_score:.6f} -> {val_score:.6f}). Saving model..."
                )
                self.best_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

        print(f"Training finished. Best Validation Score: {self.best_score:.6f}")

    def generate_submission(self):
        """
        Generates predictions for the test set and saves the submission CSV.
        """
        print("\nGenerating Submission...")

        # Load Best Model
        if not os.path.exists(self.best_model_path):
            print("No best model found. Skipping submission.")
            return

        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        # Prepare Test Dataset
        test_dataset = GNSSSequenceDataset(
            split="test",
            scaler=self.train_dataset.scaler,
            load_cached_data=self.load_cached_data,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        results = {
            "tripId": [],
            "UnixTimeMillis": [],
            "LatitudeDegrees": [],
            "LongitudeDegrees": [],
        }

        with torch.no_grad():
            for X, metadata in test_loader:
                X = X.to(self.device)

                # Forward pass
                preds = self.model(X)

                # Get high-res prediction (scale 0) -> (B, 2, L)
                # Output is [Delta East, Delta North]
                pred_deltas = preds[0].cpu().numpy()

                # Metadata
                pad_mask = metadata["pad_mask"].numpy().astype(bool)
                wls_lat = metadata["WlsLatitudeDegrees"].numpy()
                wls_lon = metadata["WlsLongitudeDegrees"].numpy()
                timestamps = metadata["UnixTimeMillis"].numpy()
                drive_ids = metadata["drive_id"]
                phone_names = metadata["phone_name"]

                batch_size = X.size(0)

                for i in range(batch_size):
                    # Get valid indices for this sequence
                    valid_idx = pad_mask[i]

                    if not np.any(valid_idx):
                        continue

                    # Extract valid data
                    d_east = pred_deltas[i, 0, valid_idx]
                    d_north = pred_deltas[i, 1, valid_idx]
                    ref_lat = wls_lat[i, valid_idx]
                    ref_lon = wls_lon[i, valid_idx]
                    ts = timestamps[i, valid_idx]

                    d_id = drive_ids[i]
                    p_name = phone_names[i]

                    # Reconstruct Lat/Lon from ENU offsets
                    # We assume Up offset is 0 relative to WLS baseline for horizontal correction
                    pred_lat, pred_lon, _ = enu_to_wgs84(
                        e=d_east,
                        n=d_north,
                        u=np.zeros_like(d_east),
                        ref_lat=ref_lat,
                        ref_lon=ref_lon,
                        ref_alt=np.zeros_like(
                            ref_lat
                        ),  # Approx relative to WLS surface
                    )

                    # Construct Trip ID
                    # Format: drive_id-phone_name (e.g., 2020-05-15-US-MTV-1-Pixel4)
                    # Note: drive_id already contains dashes.
                    trip_ids = [f"{d_id}-{p_name}"] * len(ts)

                    results["tripId"].extend(trip_ids)
                    results["UnixTimeMillis"].extend(ts)
                    results["LatitudeDegrees"].extend(pred_lat)
                    results["LongitudeDegrees"].extend(pred_lon)

        # Create DataFrame
        submission_df = pd.DataFrame(results)

        # Sort by tripId and Time
        submission_df = submission_df.sort_values(by=["tripId", "UnixTimeMillis"])

        # Save
        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
        print(f"Total predictions: {len(submission_df)}")
