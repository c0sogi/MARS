import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from library.config import Config
from library.utils import TargetScaler, seed_everything
from library.data_processing import process_and_cache_features
from library.dataset import VolcanoDataset
from library.model import ChannelAdaptiveHybridModel


class Trainer:
    """
    Manages the training and validation lifecycle of the Channel-Adaptive Hybrid Model.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        target_scaler: TargetScaler,
        num_stats_features: int,
    ):
        """
        Args:
            train_loader (DataLoader): Loader for training data.
            val_loader (DataLoader): Loader for validation data.
            target_scaler (TargetScaler): Fitted scaler for inverse transforming predictions.
            num_stats_features (int): Dimension of the statistical feature vector.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.target_scaler = target_scaler
        self.device = Config.DEVICE

        # Initialize Model
        self.model = ChannelAdaptiveHybridModel(num_stats_features=num_stats_features)
        self.model.to(self.device)

        # Optimization
        self.criterion = nn.L1Loss()  # MAE Loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

    def train_epoch(self) -> float:
        """
        Runs one epoch of training.

        Returns:
            float: Average training loss (scaled MAE).
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = len(self.train_loader.dataset)

        for spectrogram, stats, target in self.train_loader:
            spectrogram = spectrogram.to(self.device)
            stats = stats.to(self.device)
            target = target.to(self.device)

            # Reshape target to match model output [Batch, 1]
            target = target.view(-1, 1)

            self.optimizer.zero_grad()

            output = self.model(spectrogram, stats)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * spectrogram.size(0)

        return running_loss / dataset_size

    def validate(self):
        """
        Runs validation on the validation set.

        Returns:
            tuple: (avg_val_loss_scaled, avg_val_mae_original)
        """
        self.model.eval()
        running_loss = 0.0
        running_mae_orig = 0.0
        dataset_size = len(self.val_loader.dataset)

        with torch.no_grad():
            for spectrogram, stats, target in self.val_loader:
                spectrogram = spectrogram.to(self.device)
                stats = stats.to(self.device)
                target = target.to(self.device)

                target_view = target.view(-1, 1)

                output = self.model(spectrogram, stats)

                # 1. Scaled Loss (for optimization/scheduler)
                loss = self.criterion(output, target_view)
                running_loss += loss.item() * spectrogram.size(0)

                # 2. Original Scale MAE (for reporting)
                # Convert to numpy and flatten
                pred_np = output.cpu().numpy().flatten()
                target_np = target.cpu().numpy().flatten()

                # Inverse transform
                pred_orig = self.target_scaler.inverse_transform(pred_np)
                target_orig = self.target_scaler.inverse_transform(target_np)

                # Calculate absolute error in original units
                mae_orig = np.abs(pred_orig - target_orig).sum()
                running_mae_orig += mae_orig

        avg_loss = running_loss / dataset_size
        avg_mae_orig = running_mae_orig / dataset_size

        return avg_loss, avg_mae_orig

    def fit(self):
        """
        Executes the training pipeline with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_loss, val_mae_orig = self.validate()

            # Step Scheduler
            self.scheduler.step(val_loss)

            epoch_time = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch + 1}/{Config.EPOCHS} | "
                f"Time: {epoch_time:.4f}s | "
                f"Train Loss (Scaled): {train_loss} | "
                f"Val Loss (Scaled): {val_loss} | "
                f"Val MAE (Original): {val_mae_orig}"
            )

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print(f"Validation loss improved. Model saved to {Config.MODEL_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break


def run_training(load_cached_data: bool = True):
    """
    Orchestrates the data preparation and training process.

    Args:
        load_cached_data (bool): Whether to use cached parquet files for features.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)

    print("--- Data Preparation ---")
    # 2. Load Metadata
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA)
    df_val_meta = pd.read_csv(Config.VAL_METADATA)

    # Debugging: Subset for faster iteration if enabled
    if Config.DEBUG:
        print(f"DEBUG MODE: Using {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train_meta = df_train_meta.iloc[: Config.DEBUG_SAMPLE_SIZE]
        df_val_meta = df_val_meta.iloc[: Config.DEBUG_SAMPLE_SIZE]

    # 3. Process Statistical Features
    # Note: We process features for the specific metadata split
    # If caching is enabled, it checks the path defined in Config
    df_train_stats = process_and_cache_features(
        Config.TRAIN_METADATA, Config.TRAIN_FEATURES_PATH, load_cached_data
    )
    df_val_stats = process_and_cache_features(
        Config.VAL_METADATA, Config.VAL_FEATURES_PATH, load_cached_data
    )

    # Filter stats to match current metadata (in case cache is full but we are debugging)
    if Config.DEBUG:
        df_train_stats = df_train_stats[
            df_train_stats["segment_id"].isin(df_train_meta["segment_id"])
        ]
        df_val_stats = df_val_stats[
            df_val_stats["segment_id"].isin(df_val_meta["segment_id"])
        ]

    # 4. Fit and Save Scalers
    print("Fitting scalers...")

    # A. Target Scaler
    target_scaler = TargetScaler()
    target_scaler.fit(df_train_meta["time_to_eruption"].values)
    target_scaler.save(Config.TARGET_MEAN_PATH, Config.TARGET_STD_PATH)

    # B. Stats Scaler
    # Identify feature columns (exclude ID and target)
    feature_cols = sorted(
        [
            c
            for c in df_train_stats.columns
            if c not in ["segment_id", "time_to_eruption"]
        ]
    )

    stats_scaler = StandardScaler()
    stats_scaler.fit(df_train_stats[feature_cols].values)

    # Save Stats Scaler manually
    os.makedirs(os.path.dirname(Config.STATS_SCALER_MEAN_PATH), exist_ok=True)
    np.save(Config.STATS_SCALER_MEAN_PATH, stats_scaler.mean_)
    np.save(Config.STATS_SCALER_SCALE_PATH, stats_scaler.scale_)

    # 5. Create Datasets and Loaders
    print("Creating Datasets...")
    train_dataset = VolcanoDataset(
        metadata_df=df_train_meta,
        stats_df=df_train_stats,
        augment=True,  # Apply SpecAugment
        target_scaler=target_scaler,
        stats_scaler=stats_scaler,
    )

    val_dataset = VolcanoDataset(
        metadata_df=df_val_meta,
        stats_df=df_val_stats,
        augment=False,
        target_scaler=target_scaler,
        stats_scaler=stats_scaler,
    )

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

    # 6. Initialize and Run Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        target_scaler=target_scaler,
        num_stats_features=len(feature_cols),
    )

    trainer.fit()
