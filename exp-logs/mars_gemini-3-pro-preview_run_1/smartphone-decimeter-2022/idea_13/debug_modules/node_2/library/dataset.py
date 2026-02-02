import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.data_processing import GNSSPreprocessor


class GNSSSequenceDataset(Dataset):
    def __init__(
        self,
        df,
        feature_cols,
        target_cols=None,
        window_size=256,
        stride=128,
        mode="train",
        stats=None,
    ):
        """
        PyTorch Dataset for GNSS sequences.

        Args:
            df (pd.DataFrame): DataFrame containing the processed sensor data.
            feature_cols (list): List of feature column names to use.
            target_cols (list): List of target column names (optional).
            window_size (int): Length of the time sequence window.
            stride (int): Step size for the sliding window.
            mode (str): 'train', 'val', or 'test'.
            stats (dict): Normalization statistics {'mean': ..., 'std': ...}.
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.window_size = window_size
        self.stride = stride
        self.mode = mode

        # Ensure tripId exists for grouping
        if "tripId" not in self.df.columns:
            self.df["tripId"] = self.df["drive_id"] + "-" + self.df["phone_name"]

        self.trip_ids = self.df["tripId"].unique()

        # Generate list of (trip_id, start_index) tuples for all windows
        self.samples = []
        self.trip_data = {}

        for trip in self.trip_ids:
            # Sort by time to ensure sequence order
            trip_df = (
                self.df[self.df["tripId"] == trip]
                .sort_values("UnixTimeMillis")
                .reset_index(drop=True)
            )
            self.trip_data[trip] = trip_df

            num_rows = len(trip_df)

            # Generate sliding windows
            # For inference (val/test), we ensure we cover the whole sequence without redundancy if stride=window_size
            for start_idx in range(0, num_rows, self.stride):
                # If we are in training, we might drop the last partial window if it's too small,
                # but here we pad everything to keep logic consistent and maximize data usage.
                self.samples.append((trip, start_idx))

        # Compute or assign normalization statistics
        if stats is None and mode == "train":
            # Compute stats on the whole training dataframe
            print("Computing normalization statistics on training data...")
            feats = self.df[self.feature_cols].values.astype(np.float32)
            self.stats = {
                "mean": np.nanmean(feats, axis=0),
                "std": np.nanstd(feats, axis=0)
                + 1e-8,  # Add epsilon to avoid division by zero
            }
        else:
            self.stats = stats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        trip_id, start_idx = self.samples[idx]
        trip_df = self.trip_data[trip_id]

        end_idx = start_idx + self.window_size

        # Extract sequence chunk
        chunk = trip_df.iloc[start_idx:end_idx]

        # Prepare Features
        features = chunk[self.feature_cols].values.astype(np.float32)

        # Apply Normalization
        if self.stats is not None:
            features = (features - self.stats["mean"]) / self.stats["std"]

        # Handle Padding if the chunk is shorter than window_size
        seq_len = features.shape[0]
        pad_len = self.window_size - seq_len

        if pad_len > 0:
            # Pad with zeros at the end
            # shape is (seq_len, num_features), pad width ((top, bottom), (left, right))
            features = np.pad(
                features, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
            )

        # Transpose to (Channels, Time) for PyTorch Conv1d layers
        # Input: (Time, Channels) -> Output: (Channels, Time)
        features = features.transpose(1, 0)

        # Construct output dictionary
        item = {
            "features": torch.tensor(features, dtype=torch.float32),
            "trip_id": trip_id,
            "start_idx": start_idx,
            "seq_len": seq_len,  # Useful for masking padding during loss/inference
        }

        # Prepare Targets (Training/Validation only)
        if self.target_cols and self.mode != "test":
            targets = chunk[self.target_cols].values.astype(np.float32)

            if pad_len > 0:
                targets = np.pad(
                    targets, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
                )

            # Transpose targets to (Channels, Time)
            targets = targets.transpose(1, 0)
            item["targets"] = torch.tensor(targets, dtype=torch.float32)

            # Create a mask for valid time steps (1 for valid, 0 for padded)
            mask = np.ones((self.window_size,), dtype=np.float32)
            if pad_len > 0:
                mask[seq_len:] = 0
            item["mask"] = torch.tensor(mask, dtype=torch.float32)

        # Add metadata needed for reconstruction (Lat/Lon recovery)
        # We pad these as well to maintain batch shape consistency
        timestamps = chunk["UnixTimeMillis"].values
        if pad_len > 0:
            timestamps = np.pad(
                timestamps, (0, pad_len), mode="constant", constant_values=0
            )
        item["UnixTimeMillis"] = timestamps

        if "wls_lat" in chunk.columns:
            wls_lat = chunk["wls_lat"].values
            wls_lon = chunk["wls_lon"].values
            if pad_len > 0:
                wls_lat = np.pad(
                    wls_lat, (0, pad_len), mode="constant", constant_values=0
                )
                wls_lon = np.pad(
                    wls_lon, (0, pad_len), mode="constant", constant_values=0
                )
            item["wls_lat"] = wls_lat
            item["wls_lon"] = wls_lon

        return item


def get_dataloaders(debug=False):
    """
    Creates DataLoaders for training and validation.

    Args:
        debug (bool): If True, uses a smaller subset of data for debugging.

    Returns:
        train_loader, val_loader, stats
    """
    preprocessor = GNSSPreprocessor()

    # Load processed dataframes
    # The preprocessor handles caching internally
    train_df = preprocessor.get_train_data(load_cached_data=True, debug=debug)
    val_df = preprocessor.get_val_data(load_cached_data=True, debug=debug)

    feature_cols = Config.FEATURE_NAMES
    # Targets are delta meters from WLS baseline
    target_cols = ["d_east", "d_north"]

    # Instantiate Train Dataset
    # Use overlap (stride < window_size) for training data augmentation
    train_dataset = GNSSSequenceDataset(
        train_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window_size=Config.TRAIN_WINDOW_SIZE,
        stride=Config.TRAIN_WINDOW_SIZE // 2,
        mode="train",
    )

    # Instantiate Validation Dataset
    # Use non-overlapping windows for validation to evaluate each point exactly once (mostly)
    val_dataset = GNSSSequenceDataset(
        val_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window_size=Config.TRAIN_WINDOW_SIZE,
        stride=Config.TRAIN_WINDOW_SIZE,
        mode="val",
        stats=train_dataset.stats,  # Use training stats for normalization
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches during training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, train_dataset.stats


def get_test_dataloader(stats, debug=False):
    """
    Creates a DataLoader for the test set.

    Args:
        stats (dict): Normalization statistics from the training set.
        debug (bool): If True, uses a subset.

    Returns:
        test_loader
    """
    preprocessor = GNSSPreprocessor()
    test_df = preprocessor.get_test_data(load_cached_data=True, debug=debug)

    feature_cols = Config.FEATURE_NAMES

    test_dataset = GNSSSequenceDataset(
        test_df,
        feature_cols=feature_cols,
        target_cols=None,  # No targets in test
        window_size=Config.TRAIN_WINDOW_SIZE,
        stride=Config.TRAIN_WINDOW_SIZE,  # Non-overlapping for inference
        mode="test",
        stats=stats,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return test_loader
