import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from library.config import Config
from library.utils import get_logger, WGS84
from library.model import SKFNet
from library.dataset import SKFDataset


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the SKFNet model.
    """

    def __init__(self):
        self.logger = get_logger("trainer")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = SKFNet().to(self.device)
        self.criterion = nn.L1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.SCHEDULER_MIN_LR,
            verbose=True,
        )

        self.best_val_loss = float("inf")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (x_seq, x_sky, y) in enumerate(train_loader):
            x_seq = x_seq.to(self.device)
            x_sky = x_sky.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(x_seq, x_sky)
            loss = self.criterion(outputs, y)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * x_seq.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader: DataLoader) -> float:
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for x_seq, x_sky, y in val_loader:
                x_seq = x_seq.to(self.device)
                x_sky = x_sky.to(self.device)
                y = y.to(self.device)

                outputs = self.model(x_seq, x_sky)
                loss = self.criterion(outputs, y)

                running_loss += loss.item() * x_seq.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        return epoch_loss

    def fit(self, train_dataset: SKFDataset, val_dataset: SKFDataset):
        """
        Runs the full training loop with early stopping.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(
            f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}"
        )

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.scheduler.step(val_loss)

            self.logger.info(
                f"Epoch {epoch+1}/{Config.EPOCHS} - "
                f"Train Loss (MAE): {train_loss:.9f} - "
                f"Val Loss (MAE): {val_loss:.9f}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                self.logger.info(f"New best model saved to {Config.MODEL_PATH}")
            else:
                patience_counter += 1
                self.logger.info(
                    f"Early stopping counter: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training complete. Best Val Loss: {self.best_val_loss:.9f}")

    def predict(self, test_dataset: SKFDataset, test_meta: pd.DataFrame):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            test_dataset: The preprocessed test dataset.
            test_meta: DataFrame containing metadata (WLS positions) for reconstruction.
        """
        # Load best model
        if os.path.exists(Config.MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_PATH, map_location=self.device)
            )
            self.logger.info(f"Loaded best model from {Config.MODEL_PATH}")
        else:
            self.logger.warning(
                "No checkpoint found. Using current model state (random or last epoch)."
            )

        self.model.eval()

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        predictions = []

        self.logger.info("Running inference on test set...")
        with torch.no_grad():
            for x_seq, x_sky, _ in tqdm(test_loader, desc="Inference"):
                x_seq = x_seq.to(self.device)
                x_sky = x_sky.to(self.device)

                # Output: (Batch, 2) -> [DeltaEast, DeltaNorth] in meters
                outputs = self.model(x_seq, x_sky)
                predictions.append(outputs.cpu().numpy())

        # Concatenate all batch predictions
        pred_residuals = np.concatenate(predictions, axis=0)

        # Reconstruction
        # Target was: GroundTruth - WLS
        # So: Prediction = WLS + PredictedResidual
        # We need to convert metric residuals back to degrees

        self.logger.info("Reconstructing coordinates...")

        # Ensure metadata aligns with dataset
        if len(test_meta) != len(pred_residuals):
            raise ValueError(
                f"Metadata length ({len(test_meta)}) does not match predictions ({len(pred_residuals)})"
            )

        wls_lats = test_meta["lat"].values
        wls_lons = test_meta["lon"].values

        d_east = pred_residuals[:, 0]
        d_north = pred_residuals[:, 1]

        # Convert meters to degrees
        d_lat_deg, d_lon_deg = WGS84.meters_to_lat_lon_flat(d_north, d_east, wls_lats)

        pred_lats = wls_lats + d_lat_deg
        pred_lons = wls_lons + d_lon_deg

        # Prepare submission dataframe
        submission_df = pd.DataFrame(
            {
                "tripId": test_meta["tripId"],
                "UnixTimeMillis": test_meta["UnixTimeMillis"],
                "LatitudeDegrees": pred_lats,
                "LongitudeDegrees": pred_lons,
            }
        )

        # Save submission
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
