import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import INPUT_DIR, N_PULSES
from library.utils import load_sensor_geometry


class IceCubeDataset(Dataset):
    """
    PyTorch Dataset for IceCube Neutrino Direction Prediction.

    Handles loading of event metadata, batch parquet files, and sensor geometry.
    Performs pulse selection, normalization, and padding.
    """

    def __init__(
        self, metadata_path: str, mode: str = "train", debug_subset_size: int = None
    ):
        """
        Args:
            metadata_path (str): Path to the parquet metadata file.
            mode (str): 'train', 'val', or 'test'. Determines if targets are returned.
            debug_subset_size (int, optional): If set, limits the dataset size for debugging.
        """
        self.metadata_path = metadata_path
        self.mode = mode
        self.debug_subset_size = debug_subset_size

        # Load Sensor Geometry
        # Returns DF with index 'sensor_id' and columns ['x', 'y', 'z']
        self.geometry = load_sensor_geometry()

        # Load Metadata
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        self.meta_df = pd.read_parquet(metadata_path)

        # Debugging: Subset
        if self.debug_subset_size is not None and self.debug_subset_size < len(
            self.meta_df
        ):
            self.meta_df = self.meta_df.iloc[: self.debug_subset_size].copy()

        # Reset index to ensure 0..N-1 access
        self.meta_df = self.meta_df.reset_index(drop=True)

        # Batch Caching Mechanism
        # We cache the dataframe of the currently loaded batch file to reduce I/O
        self.current_batch_df = None
        self.current_batch_id = -1

        # Normalization Constants (derived from data analysis)
        self.norm_stats = {
            "time_mean": 13000.0,
            "time_std": 4500.0,
            "coord_scale": 300.0,
        }

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        """
        Returns:
            features (torch.Tensor): Shape (N_PULSES, 6).
                                     Channels: [x, y, z, time, charge, auxiliary]
            target (torch.Tensor): Shape (3,). Vector (nx, ny, nz).
                                   Returns zeros for 'test' mode.
        """
        # 1. Get Event Metadata
        row = self.meta_df.iloc[idx]
        batch_id = row["batch_id"]
        event_id = row["event_id"]

        # 2. Load Batch Data (with caching)
        if self.current_batch_id != batch_id:
            batch_rel_path = row["batch_file_path"]
            full_batch_path = os.path.join(INPUT_DIR, batch_rel_path)

            if not os.path.exists(full_batch_path):
                raise FileNotFoundError(f"Batch file missing: {full_batch_path}")

            # Load the batch file
            # Note: The batch files have event_id as index usually, but we need to check format
            # Based on description: "event_id (int): ... Saved as the index column in parquet."
            # We read it and ensure we can access by integer location or index
            self.current_batch_df = pd.read_parquet(full_batch_path)
            self.current_batch_id = batch_id

        # 3. Extract Pulses for this Event
        # Metadata contains first_pulse_index and last_pulse_index relative to the batch file
        first_idx = int(row["first_pulse_index"])
        last_idx = int(row["last_pulse_index"])

        # Slice the dataframe (inclusive of last_idx implies we need +1 for python slice)
        # We use iloc for speed as we have integer positions
        event_pulses = self.current_batch_df.iloc[first_idx : last_idx + 1].copy()

        # 4. Merge Geometry
        # event_pulses has 'sensor_id', 'time', 'charge', 'auxiliary'
        # geometry has 'x', 'y', 'z' indexed by 'sensor_id'
        # We perform a left join on sensor_id
        event_pulses = event_pulses.merge(
            self.geometry, left_on="sensor_id", right_index=True, how="left"
        )

        # Handle missing geometry (unlikely, but safe to fill 0)
        event_pulses[["x", "y", "z"]] = event_pulses[["x", "y", "z"]].fillna(0.0)

        # 5. Pulse Selection (Top N by Charge)
        # Sort by charge descending
        if len(event_pulses) > N_PULSES:
            event_pulses = event_pulses.sort_values(by="charge", ascending=False).iloc[
                :N_PULSES
            ]

        # 6. Feature Extraction & Normalization
        # Features: [x, y, z, time, charge, auxiliary]

        # Coordinates
        x = event_pulses["x"].values / self.norm_stats["coord_scale"]
        y = event_pulses["y"].values / self.norm_stats["coord_scale"]
        z = event_pulses["z"].values / self.norm_stats["coord_scale"]

        # Time
        t = (
            event_pulses["time"].values - self.norm_stats["time_mean"]
        ) / self.norm_stats["time_std"]

        # Charge (Log transform)
        q = np.log1p(event_pulses["charge"].values)

        # Auxiliary (Boolean to float)
        aux = event_pulses["auxiliary"].values.astype(np.float32)

        # Stack features
        # Shape: (num_pulses, 6)
        features_np = np.stack([x, y, z, t, q, aux], axis=1).astype(np.float32)

        # 7. Padding
        num_pulses = features_np.shape[0]
        if num_pulses < N_PULSES:
            pad_size = N_PULSES - num_pulses
            padding = np.zeros((pad_size, 6), dtype=np.float32)
            features_np = np.concatenate([features_np, padding], axis=0)

        features_tensor = torch.from_numpy(features_np)

        # 8. Target Generation
        target_tensor = torch.zeros(3, dtype=torch.float32)

        if self.mode != "test":
            azimuth = float(row["azimuth"])
            zenith = float(row["zenith"])

            # Convert spherical to cartesian unit vector
            # x = cos(az) * sin(ze)
            # y = sin(az) * sin(ze)
            # z = cos(ze)

            tx = np.cos(azimuth) * np.sin(zenith)
            ty = np.sin(azimuth) * np.sin(zenith)
            tz = np.cos(zenith)

            target_tensor = torch.tensor([tx, ty, tz], dtype=torch.float32)

        return features_tensor, target_tensor
