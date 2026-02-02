import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.features import process_dataset
from library.utils import get_logger

logger = get_logger("dataset")


class SmartphoneLocationDataset(Dataset):
    """
    PyTorch Dataset for Smartphone Decimeter Challenge.
    Handles sequence slicing, normalization, and target preparation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mode: str = "train",
        scaler: dict = None,
        window_size: int = 128,
        stride: int = 64,
    ):
        """
        Args:
            df: DataFrame containing processed features and metadata.
            mode: 'train', 'val', or 'test'.
            scaler: Dictionary containing 'mean' and 'std' for normalization.
            window_size: Length of the time sequence window.
            stride: Step size for sliding window.
        """
        self.mode = mode
        self.window_size = window_size
        self.stride = stride

        # Extract features
        self.features = df[Config.FEATURE_COLS].values.astype(np.float32)

        # Normalize features
        if scaler is not None:
            self.features = (self.features - scaler["mean"]) / (scaler["std"] + 1e-8)
            self.features = self.features.astype(np.float32)

        # Extract targets for train/val
        if self.mode in ["train", "val"]:
            self.targets = df[Config.TARGET_COLS].values.astype(np.float32)

        # Keep metadata for reconstruction
        self.meta_cols = [
            "drive_id",
            "phone_name",
            "UnixTimeMillis",
            "WlsLatitudeDegrees",
            "WlsLongitudeDegrees",
        ]
        # Ensure columns exist (WLS cols are added by process_dataset)
        for col in self.meta_cols:
            if col not in df.columns:
                df[col] = 0
        self.meta = df[self.meta_cols].copy()

        # Generate sequence indices
        self.indices = self._prepare_sequences(df)

    def _prepare_sequences(self, df):
        """
        Slices the dataframe into sequences grouped by drive and phone.
        """
        indices = []
        # Group by drive_id and phone_name to respect trip boundaries
        # Using groupby indices is efficient
        groups = df.groupby(["drive_id", "phone_name"]).indices

        for _, group_idxs in groups.items():
            group_idxs = np.sort(group_idxs)
            n_samples = len(group_idxs)

            # For inference (test/val), we want to cover the whole sequence without overlap if possible,
            # or with minimal overlap. For training, we want overlap (stride < window).
            # Config defines TRAIN parameters. We override stride for evaluation if needed.
            current_stride = self.stride

            # Generate windows
            for start_idx in range(0, n_samples, current_stride):
                end_idx = min(start_idx + self.window_size, n_samples)
                seq_len = end_idx - start_idx

                # We keep the slice indices and the actual length
                indices.append((group_idxs[start_idx:end_idx], seq_len))

                # If we reached the end, break
                if end_idx == n_samples:
                    break

        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Retrieve indices for this sequence
        global_indices, seq_len = self.indices[idx]

        # Get features
        x = self.features[global_indices]  # Shape: (seq_len, input_dim)

        # Prepare padding
        pad_len = self.window_size - seq_len

        # Create mask (1 for data, 0 for padding)
        mask = np.ones(self.window_size, dtype=np.float32)

        if pad_len > 0:
            # Pad features with 0
            x = np.pad(x, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)
            mask[seq_len:] = 0

        item = {"features": torch.from_numpy(x), "mask": torch.from_numpy(mask)}

        # Handle Targets
        if self.mode in ["train", "val"]:
            y = self.targets[global_indices]  # Shape: (seq_len, output_dim)
            if pad_len > 0:
                y = np.pad(
                    y, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
                )
            item["targets"] = torch.from_numpy(y)

        # Handle Metadata (Essential for WLS correction)
        # We return the metadata for the valid part of the sequence, padded to window size
        # This allows batch processing of metadata if needed

        # Extract WLS and Timestamp
        wls_lat = self.meta.iloc[global_indices]["WlsLatitudeDegrees"].values.astype(
            np.float64
        )
        wls_lon = self.meta.iloc[global_indices]["WlsLongitudeDegrees"].values.astype(
            np.float64
        )
        timestamps = self.meta.iloc[global_indices]["UnixTimeMillis"].values.astype(
            np.int64
        )

        if pad_len > 0:
            # Pad metadata with edge values or 0. Edge is safer for coordinate transforms to avoid singularities.
            wls_lat = np.pad(wls_lat, (0, pad_len), mode="edge")
            wls_lon = np.pad(wls_lon, (0, pad_len), mode="edge")
            timestamps = np.pad(timestamps, (0, pad_len), mode="edge")

        item["wls_lat"] = torch.from_numpy(wls_lat)
        item["wls_lon"] = torch.from_numpy(wls_lon)
        item["timestamps"] = torch.from_numpy(timestamps)

        return item


def get_train_val_loaders(
    batch_size=Config.BATCH_SIZE, load_cached_data=True, debug=False
):
    """
    Loads train and validation data, computes scaler, and returns DataLoaders.

    Args:
        batch_size: Batch size for training.
        load_cached_data: Whether to use cached parquet files.
        debug: If True, subsets data for quick debugging.

    Returns:
        train_loader, val_loader, scaler (dict)
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Load Dataframes using features module
    logger.info("Loading Training Data...")
    train_df = process_dataset(mode="train", load_cached_data=load_cached_data)

    logger.info("Loading Validation Data...")
    val_df = process_dataset(mode="val", load_cached_data=load_cached_data)

    if debug:
        logger.info("Debug mode: Subsetting data...")
        train_df = train_df.iloc[:10000]
        val_df = val_df.iloc[:2000]

    # Compute Scaler from Training Data
    logger.info("Computing Feature Scaler...")
    features = train_df[Config.FEATURE_COLS].values
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    # Avoid division by zero
    std[std == 0] = 1.0

    scaler = {"mean": mean, "std": std}

    # Create Datasets
    # Train: Overlapping windows for data augmentation
    train_dataset = SmartphoneLocationDataset(
        train_df,
        mode="train",
        scaler=scaler,
        window_size=Config.TRAIN_WINDOW_SIZE,
        stride=Config.TRAIN_WINDOW_STRIDE,
    )

    # Val: Non-overlapping windows for efficient evaluation
    val_dataset = SmartphoneLocationDataset(
        val_df,
        mode="val",
        scaler=scaler,
        window_size=Config.TRAIN_WINDOW_SIZE,
        stride=Config.TRAIN_WINDOW_SIZE,  # Non-overlapping
    )

    # Create Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(f"Train Batches: {len(train_loader)}, Val Batches: {len(val_loader)}")

    return train_loader, val_loader, scaler


def get_test_loader(scaler, batch_size=Config.BATCH_SIZE, load_cached_data=True):
    """
    Loads test data and returns DataLoader.

    Args:
        scaler: Normalization stats from training.
        batch_size: Batch size for inference.
        load_cached_data: Whether to use cached parquet files.

    Returns:
        test_loader
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    logger.info("Loading Test Data...")
    test_df = process_dataset(mode="test", load_cached_data=load_cached_data)

    # Create Dataset
    # Test: Non-overlapping windows to cover sequence exactly once (tiling)
    test_dataset = SmartphoneLocationDataset(
        test_df,
        mode="test",
        scaler=scaler,
        window_size=Config.TRAIN_WINDOW_SIZE,
        stride=Config.TRAIN_WINDOW_SIZE,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(f"Test Batches: {len(test_loader)}")

    return test_loader
