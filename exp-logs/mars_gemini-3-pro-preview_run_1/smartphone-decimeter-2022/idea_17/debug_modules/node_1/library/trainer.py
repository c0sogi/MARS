import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger, fix_seed, WGS84Utils
from library.model import CascadedResUNet
from library.loss import CascadedDeepSupervisionLoss
from library.dataset import get_train_val_loaders, get_test_loader

logger = get_logger("trainer")


class Trainer:
    def __init__(self, run_name="idea_17"):
        self.run_name = run_name
        self.device = torch.device(Config.DEVICE)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Ensure reproducibility
        fix_seed(Config.SEED)

        # Initialize Model
        self.model = CascadedResUNet().to(self.device)

        # Loss Function
        self.criterion = CascadedDeepSupervisionLoss().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=1e-6
        )

    def train_epoch(self, loader, epoch):
        self.model.train()
        running_loss = 0.0
        running_final_loss = 0.0
        running_aux_loss = 0.0

        # Use tqdm for progress tracking if running interactively, otherwise silent
        pbar = tqdm(
            loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}", leave=False, disable=True
        )

        for batch in loader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Forward pass
            # Transpose features to (B, C, L) for Conv1d
            features = features.transpose(1, 2)
            targets = targets.transpose(1, 2)

            # Model returns (final_output, [aux_outputs])
            predictions = self.model(features)

            # Compute Loss
            loss, metrics = self.criterion(predictions, targets, mask)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.GRAD_CLIP_MAX_NORM
            )

            self.optimizer.step()

            # Statistics
            running_loss += metrics["loss"]
            running_final_loss += metrics["loss_final"]
            running_aux_loss += metrics["loss_aux"]

            pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})
            pbar.update()

        pbar.close()

        n_batches = len(loader)
        return {
            "loss": running_loss / n_batches,
            "loss_final": running_final_loss / n_batches,
            "loss_aux": running_aux_loss / n_batches,
        }

    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0

        # For metric calculation
        all_errors = []

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)

                # Metadata for reconstruction
                wls_lat = batch["wls_lat"].numpy()
                wls_lon = batch["wls_lon"].numpy()

                # Ground truth for metric
                # We need to reconstruct GT Lat/Lon from targets to be precise,
                # or just use the targets (dEast, dNorth) vs predictions (dEast, dNorth) distance.
                # Since target is (dEast, dNorth) in meters, L2 norm is the distance error.

                features = features.transpose(1, 2)
                targets = targets.transpose(1, 2)

                # Forward
                predictions = self.model(features)
                final_pred = predictions[0]  # (B, 2, L)

                # Loss
                loss, _ = self.criterion(predictions, targets, mask)
                running_loss += loss.item()

                # Metric Calculation: Euclidean distance between pred and target offsets
                # Shape: (B, 2, L)
                # We only care about valid points defined by mask
                valid_mask = mask.bool().cpu().numpy()  # (B, L)

                pred_np = final_pred.cpu().numpy()
                target_np = targets.cpu().numpy()

                # Calculate error in meters for valid points
                # Errors = sqrt((pred_e - target_e)^2 + (pred_n - target_n)^2)
                diff = pred_np - target_np
                dist_errors = np.sqrt(np.sum(diff**2, axis=1))  # (B, L)

                # Flatten and filter by mask
                valid_errors = dist_errors[valid_mask]
                all_errors.extend(valid_errors)

        # Aggregate Metrics
        all_errors = np.array(all_errors)
        mean_dist = np.mean(all_errors)
        p50 = np.percentile(all_errors, 50)
        p95 = np.percentile(all_errors, 95)
        score = (p50 + p95) / 2

        return {
            "loss": running_loss / len(loader),
            "mean_dist": mean_dist,
            "p50": p50,
            "p95": p95,
            "score": score,
        }

    def fit(self, debug=False):
        logger.info(f"Starting training run: {self.run_name}")

        # Load Data
        train_loader, val_loader, scaler = get_train_val_loaders(
            batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=debug
        )

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {epoch_time:.1f}s | LR: {current_lr:.2e} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Score: {val_metrics['score']:.4f} (p50: {val_metrics['p50']:.4f}, p95: {val_metrics['p95']:.4f})"
            )

            # Checkpointing & Early Stopping
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.CHECKPOINT_PATH)
                logger.info(
                    f"  >>> New Best Model Saved (Loss: {self.best_val_loss:.4f})"
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    logger.info("Early stopping triggered.")
                    break

        return scaler

    def predict(self, scaler):
        logger.info("Starting prediction on test set...")

        # Load best model
        if not os.path.exists(Config.CHECKPOINT_PATH):
            logger.warning(
                "No checkpoint found! Using current model weights (untrained?)."
            )
        else:
            self.model.load_state_dict(
                torch.load(Config.CHECKPOINT_PATH, map_location=self.device)
            )
            logger.info("Loaded best model checkpoint.")

        self.model.eval()

        # Load Test Data
        test_loader = get_test_loader(
            scaler, batch_size=Config.BATCH_SIZE, load_cached_data=True
        )
        dataset = test_loader.dataset

        # Prepare arrays to store results in the metadata dataframe
        # Initialize with NaNs
        pred_lats = np.full(len(dataset.meta), np.nan)
        pred_lons = np.full(len(dataset.meta), np.nan)

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                features = batch["features"].to(self.device)
                mask = batch["mask"].numpy()  # (B, L)

                # Metadata
                wls_lat = batch["wls_lat"].numpy()  # (B, L)
                wls_lon = batch["wls_lon"].numpy()  # (B, L)

                # Forward
                features = features.transpose(1, 2)
                predictions = self.model(features)
                final_pred = predictions[0].cpu().numpy()  # (B, 2, L)

                dEast = final_pred[:, 0, :]
                dNorth = final_pred[:, 1, :]

                # Map back to global dataframe indices
                # The dataset returns sequences based on `self.indices` list.
                # We need to find which global indices correspond to this batch.
                start_seq_idx = batch_idx * test_loader.batch_size
                end_seq_idx = min(
                    start_seq_idx + test_loader.batch_size, len(dataset.indices)
                )

                batch_seq_infos = dataset.indices[start_seq_idx:end_seq_idx]

                for i, (global_indices, seq_len) in enumerate(batch_seq_infos):
                    # Valid length of the sequence
                    valid_len = seq_len

                    # Extract valid predictions and metadata for this sequence
                    # Remove padding
                    curr_dEast = dEast[i, :valid_len]
                    curr_dNorth = dNorth[i, :valid_len]
                    curr_wls_lat = wls_lat[i, :valid_len]
                    curr_wls_lon = wls_lon[i, :valid_len]

                    # Reconstruct Lat/Lon
                    # Vectorized conversion
                    # Note: WGS84Utils methods are scalar/simple, let's implement vectorized here or loop
                    # Given the scale, looping inside is fine or use simple approximation

                    # Vectorized approximation for speed
                    # M, N calculation
                    # Constants
                    a = 6378137.0
                    e2 = 6.69437999014e-3

                    lat_rad = np.radians(curr_wls_lat)
                    sin_lat = np.sin(lat_rad)
                    sin_sq = sin_lat**2

                    N = a / np.sqrt(1 - e2 * sin_sq)
                    M = (a * (1 - e2)) / np.power(1 - e2 * sin_sq, 1.5)

                    d_lat_rad = curr_dNorth / M
                    d_lon_rad = curr_dEast / (N * np.cos(lat_rad))

                    pred_lat = curr_wls_lat + np.degrees(d_lat_rad)
                    pred_lon = curr_wls_lon + np.degrees(d_lon_rad)

                    # Assign to global arrays
                    pred_lats[global_indices] = pred_lat
                    pred_lons[global_indices] = pred_lon

        # Assign predictions to metadata
        dataset.meta["LatitudeDegrees"] = pred_lats
        dataset.meta["LongitudeDegrees"] = pred_lons

        # Create Submission File
        # We need to merge with sample submission to ensure correct format and order
        # dataset.meta has columns: drive_id, phone_name, UnixTimeMillis, ...

        # Construct tripId
        dataset.meta["tripId"] = (
            dataset.meta["drive_id"] + "-" + dataset.meta["phone_name"]
        )

        # Load sample submission
        sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

        # Merge
        # We use left join on sample submission to ensure we have all required rows
        # and in the correct order.
        submission = sample_sub[["tripId", "UnixTimeMillis"]].merge(
            dataset.meta[
                ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
            ],
            on=["tripId", "UnixTimeMillis"],
            how="left",
            suffixes=("_old", ""),
        )

        # Fill missing predictions with WLS (if any, though shouldn't be for valid test set)
        # If model didn't predict (e.g. data processing failed for a drive), fallback?
        # Ideally, we should have predictions for everything.

        # Select final columns
        final_sub = submission[
            ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
        ]

        # Save
        os.makedirs(os.path.dirname(Config.SUBMISSION_OUTPUT_PATH), exist_ok=True)
        final_sub.to_csv(Config.SUBMISSION_OUTPUT_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_OUTPUT_PATH}")


def run():
    trainer = Trainer()
    scaler = trainer.fit(debug=Config.DEBUG)
    trainer.predict(scaler)


if __name__ == "__main__":
    run()
