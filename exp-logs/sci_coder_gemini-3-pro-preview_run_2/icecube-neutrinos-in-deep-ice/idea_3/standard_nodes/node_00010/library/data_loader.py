import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from library.config import (
    INPUT_DIR,
    SEQ_LEN,
    INPUT_CHANNELS,
    STATS,
    NUM_WORKERS,
    BATCH_SIZE,
    SENSOR_GEO_PATH,
)
from library.utils import load_geometry, angles_to_direction


class IceCubeDataset(Dataset):
    def __init__(self, metadata_path, mode="train", max_samples=None):
        """
        Args:
            metadata_path: Path to the parquet metadata file.
            mode: 'train', 'val', or 'test'.
            max_samples: Limit the number of events (for debugging/speed).
        """
        self.mode = mode
        self.seq_len = SEQ_LEN

        # Load Metadata
        # print(f"[{mode.upper()}] Loading metadata from {metadata_path}...")
        self.meta = pd.read_parquet(metadata_path)

        if max_samples is not None:
            self.meta = self.meta.iloc[:max_samples]
            # print(f"[{mode.upper()}] Limited to {len(self.meta)} samples.")

        # Load Geometry
        self.geo = load_geometry()  # Index is sensor_id, cols: x, y, z

        # Cache for Batch Data
        # We use lazy loading to avoid reading all files at initialization.
        # Cite debug_lesson_1: Avoid Eager Preloading of Large File Chunks
        self.batch_data = {}
        self.cache_capacity = 3

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        bid = row["batch_id"]

        # Lazy load batch data if not present
        if bid not in self.batch_data:
            # Cite debug_lesson_3: Avoid Unbounded Instance Caching in Multi-Worker PyTorch Datasets
            if len(self.batch_data) >= self.cache_capacity:
                self.batch_data.pop(next(iter(self.batch_data)))

            base_dir = "train" if self.mode in ["train", "val"] else "test"
            batch_file = os.path.join(INPUT_DIR, base_dir, f"batch_{bid}.parquet")

            if os.path.exists(batch_file):
                df = pd.read_parquet(batch_file)
                self.batch_data[bid] = df
            else:
                self.batch_data[bid] = pd.DataFrame()

        # Retrieve pulses using the pre-calculated indices
        # Note: These indices are relative to the start of the batch file
        f_idx = row["first_pulse_index"]
        l_idx = row["last_pulse_index"]

        if not self.batch_data[bid].empty:
            # iloc is relatively fast on in-memory dataframe
            pulses = self.batch_data[bid].iloc[f_idx : l_idx + 1].copy()
            # Map Geometry
            pulses["sensor_x"] = (
                pulses["sensor_id"].map(self.geo["x"]).astype(np.float32)
            )
            pulses["sensor_y"] = (
                pulses["sensor_id"].map(self.geo["y"]).astype(np.float32)
            )
            pulses["sensor_z"] = (
                pulses["sensor_id"].map(self.geo["z"]).astype(np.float32)
            )
            pulses.fillna(0, inplace=True)
        else:
            pulses = pd.DataFrame()

        # --- Feature Engineering ---
        if len(pulses) == 0:
            # Handle empty event
            features = np.zeros((INPUT_CHANNELS, self.seq_len), dtype=np.float32)
        else:
            # 1. Prioritize highest charge
            # We want to keep the pulses with most info.
            if len(pulses) > self.seq_len:
                pulses = pulses.sort_values(by="charge", ascending=False).iloc[
                    : self.seq_len
                ]

            # 2. Sort strictly by time for 1D CNN
            pulses = pulses.sort_values(by="time", ascending=True)

            # 3. Extract columns
            # Channels: [x, y, z, time, charge, auxiliary]
            x = pulses["sensor_x"].values
            y = pulses["sensor_y"].values
            z = pulses["sensor_z"].values
            t = pulses["time"].values
            c = pulses["charge"].values
            a = pulses["auxiliary"].values.astype(float)

            # 4. Normalize
            x = (x - STATS["x_mean"]) / STATS["x_std"]
            y = (y - STATS["y_mean"]) / STATS["y_std"]
            z = (z - STATS["z_mean"]) / STATS["z_std"]
            t = (t - STATS["time_mean"]) / STATS["time_std"]
            c = np.log1p(c)  # Log transform for charge

            # Stack features: (L, 6) -> Transpose to (6, L)
            # Pad if length < seq_len
            curr_len = len(pulses)
            features = np.zeros((INPUT_CHANNELS, self.seq_len), dtype=np.float32)

            features[0, :curr_len] = x
            features[1, :curr_len] = y
            features[2, :curr_len] = z
            features[3, :curr_len] = t
            features[4, :curr_len] = c
            features[5, :curr_len] = a

        # --- Targets ---
        if self.mode != "test":
            azimuth = row["azimuth"]
            zenith = row["zenith"]
            # Convert to vector for regression
            tx, ty, tz = angles_to_direction(azimuth, zenith)
            target = np.array([tx, ty, tz], dtype=np.float32)
            return torch.tensor(features), torch.tensor(target)
        else:
            event_id = row["event_id"]
            return torch.tensor(features), event_id


def get_dataloader(
    metadata_path,
    mode="train",
    max_samples=None,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
):
    """
    Factory function to create a DataLoader for the IceCube dataset.

    Args:
        metadata_path (str): Path to the metadata parquet file.
        mode (str): 'train', 'val', or 'test'.
        max_samples (int, optional): Limit number of samples.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.

    Returns:
        DataLoader: Configured PyTorch DataLoader.
    """
    dataset = IceCubeDataset(metadata_path, mode=mode, max_samples=max_samples)

    shuffle = mode == "train"

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return loader
