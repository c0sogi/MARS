import os
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config

# =============================================================================
# Normalization Constants (Derived from Data Analysis)
# =============================================================================
# Coordinates (approximate range scaling)
COORD_SCALE = 500.0

# Time (approximate std dev)
TIME_SCALE = 1000.0  # Scaling relative time

# Charge (Log1p transformation is used, so we scale the log values if needed,
# but usually log is enough. We'll stick to log1p)

# Global Feature Scaling
# CoG is in meters -> / 500
# Covariance is in meters^2 -> / 500^2 = / 250000
COV_SCALE = COORD_SCALE**2


class BatchLoader:
    """
    Helper class to manage loading of parquet batch files with an LRU cache.
    This avoids reloading the same batch file multiple times when workers
    access events sequentially or from the same region.
    """

    def __init__(self, capacity=2):
        self.capacity = capacity
        self.cache = {}
        self.lru_order = []

    def get_batch(self, file_path):
        # Return cached if available
        if file_path in self.cache:
            # Update LRU
            if self.lru_order[-1] != file_path:
                self.lru_order.remove(file_path)
                self.lru_order.append(file_path)
            return self.cache[file_path]

        # Load new file
        # Use pandas read_parquet - efficient for this format
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load batch file {file_path}: {e}")

        # Manage Cache
        if len(self.cache) >= self.capacity:
            oldest = self.lru_order.pop(0)
            del self.cache[oldest]

        self.cache[file_path] = df
        self.lru_order.append(file_path)

        return df


class IceCubeDataset(Dataset):
    def __init__(self, metadata_path, mode="train", transform=None, limit_size=None):
        """
        Args:
            metadata_path (str): Path to the metadata parquet file.
            mode (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
            limit_size (int, optional): Limit dataset size for debugging.
        """
        self.mode = mode
        self.transform = transform

        # 1. Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.meta = pd.read_parquet(metadata_path)

        # Debugging: Limit size
        if limit_size is not None:
            self.meta = self.meta.iloc[:limit_size].copy()

        # 2. Load Sensor Geometry
        # We map sensor_id to x, y, z using numpy arrays for fast lookup
        if not os.path.exists(Config.SENSOR_GEO_PATH):
            raise FileNotFoundError(
                f"Sensor geometry not found: {Config.SENSOR_GEO_PATH}"
            )

        geo_df = pd.read_csv(Config.SENSOR_GEO_PATH)
        max_sensor_id = geo_df["sensor_id"].max()

        # Create lookup arrays (fill with NaN or 0, though all IDs should exist)
        self.sensor_x = np.zeros(max_sensor_id + 1, dtype=np.float32)
        self.sensor_y = np.zeros(max_sensor_id + 1, dtype=np.float32)
        self.sensor_z = np.zeros(max_sensor_id + 1, dtype=np.float32)

        self.sensor_x[geo_df["sensor_id"]] = geo_df["x"].values
        self.sensor_y[geo_df["sensor_id"]] = geo_df["y"].values
        self.sensor_z[geo_df["sensor_id"]] = geo_df["z"].values

        # 3. Initialize Batch Loader
        # Each worker process will have its own instance of BatchLoader
        self.batch_loader = BatchLoader(capacity=2)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        # Retrieve metadata row
        row = self.meta.iloc[idx]
        event_id = int(row["event_id"])

        # Construct full batch path
        batch_rel_path = row["batch_file_path"]
        batch_path = os.path.join(Config.INPUT_DIR, batch_rel_path)

        # Load pulses
        batch_df = self.batch_loader.get_batch(batch_path)

        # Extract pulses for this event using integer slicing (fastest)
        # Metadata contains inclusive indices relative to the batch file
        first_idx = int(row["first_pulse_index"])
        last_idx = int(row["last_pulse_index"])

        # Slice
        pulses = batch_df.iloc[first_idx : last_idx + 1].copy()

        # Map Geometry
        sensor_ids = pulses["sensor_id"].values
        pulses["x"] = self.sensor_x[sensor_ids]
        pulses["y"] = self.sensor_y[sensor_ids]
        pulses["z"] = self.sensor_z[sensor_ids]

        # ---------------------------------------------------------------------
        # Stream 2: Geometric Features (Global)
        # Computed on ALL pulses before filtering
        # ---------------------------------------------------------------------
        charges = pulses["charge"].values.astype(np.float32)
        total_charge = np.sum(charges)

        if total_charge > 0:
            weights = charges / total_charge

            # Center of Gravity (CoG)
            cog_x = np.sum(pulses["x"].values * weights)
            cog_y = np.sum(pulses["y"].values * weights)
            cog_z = np.sum(pulses["z"].values * weights)

            # Covariance
            dx = pulses["x"].values - cog_x
            dy = pulses["y"].values - cog_y
            dz = pulses["z"].values - cog_z

            cov_xx = np.sum(weights * dx * dx)
            cov_yy = np.sum(weights * dy * dy)
            cov_zz = np.sum(weights * dz * dz)
            cov_xy = np.sum(weights * dx * dy)
            cov_xz = np.sum(weights * dx * dz)
            cov_yz = np.sum(weights * dy * dz)
        else:
            # Fallback for empty/zero charge events (rare)
            cog_x, cog_y, cog_z = 0.0, 0.0, 0.0
            cov_xx, cov_yy, cov_zz = 0.0, 0.0, 0.0
            cov_xy, cov_xz, cov_yz = 0.0, 0.0, 0.0

        # Create Geometric Feature Vector (Normalized)
        geom_feats = np.array(
            [
                cog_x / COORD_SCALE,
                cog_y / COORD_SCALE,
                cog_z / COORD_SCALE,
                cov_xx / COV_SCALE,
                cov_yy / COV_SCALE,
                cov_zz / COV_SCALE,
                cov_xy / COV_SCALE,
                cov_xz / COV_SCALE,
                cov_yz / COV_SCALE,
            ],
            dtype=np.float32,
        )

        # ---------------------------------------------------------------------
        # Stream 1: Temporal Sequence Features
        # ---------------------------------------------------------------------
        # 1. Select Top N pulses by charge
        if len(pulses) > Config.SEQ_LEN:
            # We use nlargest. Note: this sorts by charge descending
            pulses_top = pulses.nlargest(Config.SEQ_LEN, "charge")
        else:
            pulses_top = pulses

        # 2. Sort by Time (Crucial for causal convolution)
        pulses_top = pulses_top.sort_values("time")

        # 3. Extract Features
        # [x, y, z, time, charge, auxiliary]
        # Normalize
        p_x = pulses_top["x"].values / COORD_SCALE
        p_y = pulses_top["y"].values / COORD_SCALE
        p_z = pulses_top["z"].values / COORD_SCALE

        # Time: Relative to first pulse, then scaled
        p_t = pulses_top["time"].values.astype(np.float32)
        p_t = (p_t - p_t.min()) / TIME_SCALE

        # Charge: Log1p
        p_q = np.log1p(pulses_top["charge"].values).astype(np.float32)

        # Aux: Boolean to float
        p_aux = pulses_top["auxiliary"].values.astype(np.float32)

        # Stack features: Shape (Seq_Len, Channels)
        seq_features = np.stack([p_x, p_y, p_z, p_t, p_q, p_aux], axis=1)

        # 4. Pad if necessary
        curr_len = seq_features.shape[0]
        if curr_len < Config.SEQ_LEN:
            pad_len = Config.SEQ_LEN - curr_len
            # Pad with zeros
            padding = np.zeros((pad_len, Config.N_CHANNELS), dtype=np.float32)
            seq_features = np.concatenate([seq_features, padding], axis=0)

        # Transpose to (Channels, Seq_Len) for PyTorch Conv1d
        seq_features = seq_features.transpose(1, 0).astype(np.float32)

        # ---------------------------------------------------------------------
        # Targets
        # ---------------------------------------------------------------------
        if self.mode != "test":
            azimuth = row["azimuth"]
            zenith = row["zenith"]
            target = np.array([azimuth, zenith], dtype=np.float32)
        else:
            # Dummy target for test
            target = np.array([0.0, 0.0], dtype=np.float32)

        return (
            torch.tensor(seq_features),
            torch.tensor(geom_feats),
            torch.tensor(target),
            torch.tensor(event_id, dtype=torch.int64),
        )


def get_dataloaders(config):
    """
    Creates DataLoaders for training and validation.

    Args:
        config: Configuration class containing paths and params.

    Returns:
        train_loader, val_loader
    """
    # Define paths
    train_meta_path = os.path.join(Config.METADATA_DIR, "train_metadata.parquet")
    val_meta_path = os.path.join(Config.METADATA_DIR, "val_metadata.parquet")

    # Determine dataset size for debugging
    limit_size = Config.DEBUG_SUBSET_SIZE if Config.DEBUG else None

    # Create Datasets
    train_dataset = IceCubeDataset(
        metadata_path=train_meta_path, mode="train", limit_size=limit_size
    )

    val_dataset = IceCubeDataset(
        metadata_path=val_meta_path, mode="val", limit_size=limit_size
    )

    # Create DataLoaders
    # Pin memory speeds up transfer to GPU
    # Persistent workers keeps the worker processes alive, preserving the BatchLoader cache
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(config.NUM_WORKERS > 0),
    )

    return train_loader, val_loader


def get_test_dataloader(config):
    """
    Creates DataLoader for the test set.
    """
    test_meta_path = os.path.join(Config.METADATA_DIR, "test_metadata.parquet")

    test_dataset = IceCubeDataset(
        metadata_path=test_meta_path,
        mode="test",
        limit_size=None,  # Always predict full test set
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    return test_loader
