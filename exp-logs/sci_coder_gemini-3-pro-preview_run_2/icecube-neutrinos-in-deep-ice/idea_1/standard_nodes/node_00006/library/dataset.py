import os
import gc
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import (
    INPUT_DIR,
    STATS,
    SEQ_LEN,
    N_FEATURES,
    SENSOR_GEO_PATH,
)


class IceCubeDataset(Dataset):
    """
    PyTorch Dataset for IceCube Neutrino Direction Prediction.

    Handles loading pulse data from Parquet files, mapping sensor geometry,
    selecting relevant pulses, and normalizing features for the model.
    """

    def __init__(self, metadata, mode="train", cache_limit=3):
        """
        Args:
            metadata (pd.DataFrame): DataFrame containing event metadata.
            mode (str): 'train' or 'test'. If 'test', targets are dummy values.
            cache_limit (int): Number of batch files to keep in memory (LRU cache).
        """
        self.metadata = metadata.reset_index(drop=True)
        self.mode = mode
        self.cache_limit = cache_limit

        # =========================================================================
        # 1. Load and Index Sensor Geometry
        # =========================================================================
        # We create a dense numpy array for O(1) coordinate lookup:
        # sensor_map[sensor_id] -> [x, y, z]
        geo_df = pd.read_csv(SENSOR_GEO_PATH)

        # Ensure correct indexing
        if "sensor_id" in geo_df.columns:
            geo_df = geo_df.set_index("sensor_id").sort_index()

        max_sensor_id = geo_df.index.max()
        # Initialize map with zeros
        self.sensor_map = np.zeros((max_sensor_id + 1, 3), dtype=np.float32)

        # Fill the map
        coords = geo_df[["x", "y", "z"]].values
        indices = geo_df.index.values
        self.sensor_map[indices] = coords

        # =========================================================================
        # 2. Initialize Batch Cache
        # =========================================================================
        self.batch_cache = {}  # Key: batch_id, Value: DataFrame
        self.batch_order = []  # List to track usage order for LRU

    def __len__(self):
        return len(self.metadata)

    def _get_batch_df(self, batch_id, batch_file_path):
        """
        Retrieves the dataframe for a specific batch, handling caching logic.
        """
        # Hit Cache
        if batch_id in self.batch_cache:
            # Update LRU position: move to end
            self.batch_order.remove(batch_id)
            self.batch_order.append(batch_id)
            return self.batch_cache[batch_id]

        # Miss Cache: Load File
        # Enforce Cache Limit BEFORE loading new file to prevent memory spike
        while len(self.batch_cache) >= self.cache_limit:
            oldest_batch = self.batch_order.pop(0)
            del self.batch_cache[oldest_batch]
            gc.collect()  # Force garbage collection to free RAM

        full_path = os.path.join(INPUT_DIR, batch_file_path)
        df = pd.read_parquet(full_path)

        # Store in Cache
        self.batch_cache[batch_id] = df
        self.batch_order.append(batch_id)

        return df

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # ---------------------------------------------------------------------
        # 1. Retrieve Pulse Data
        # ---------------------------------------------------------------------
        batch_id = row["batch_id"]
        batch_path = row["batch_file_path"]

        # Get the full batch dataframe (cached)
        batch_df = self._get_batch_df(batch_id, batch_path)

        # Slice the specific event using provided indices
        # Metadata indices are inclusive for the event range
        p_start = row["first_pulse_index"]
        p_end = row["last_pulse_index"]

        # Slice rows. Note: iloc slice [start:end] excludes end, so we use p_end + 1
        pulses = batch_df.iloc[p_start : p_end + 1].copy()

        # ---------------------------------------------------------------------
        # 2. Map Geometry
        # ---------------------------------------------------------------------
        sensor_ids = pulses["sensor_id"].values
        # Fast lookup
        coords = self.sensor_map[sensor_ids]

        pulses["x"] = coords[:, 0]
        pulses["y"] = coords[:, 1]
        pulses["z"] = coords[:, 2]

        # ---------------------------------------------------------------------
        # 3. Pulse Selection
        # ---------------------------------------------------------------------
        # Strategy:
        # 1. Prioritize 'Good' pulses (auxiliary == False)
        # 2. Prioritize High Charge
        # Sort keys: auxiliary (asc: False=0, True=1), charge (desc)
        pulses.sort_values(
            by=["auxiliary", "charge"], ascending=[True, False], inplace=True
        )

        # Select top N pulses
        pulses = pulses.head(SEQ_LEN)

        # IMPORTANT: Re-sort strictly by time for the RNN/Sequence model
        pulses.sort_values(by="time", ascending=True, inplace=True)

        # ---------------------------------------------------------------------
        # 4. Feature Engineering & Normalization
        # ---------------------------------------------------------------------
        # Features: [x, y, z, time, charge, auxiliary]

        x = pulses["x"].values
        y = pulses["y"].values
        z = pulses["z"].values
        t = pulses["time"].values
        c = pulses["charge"].values
        a = pulses["auxiliary"].values.astype(np.float32)

        # Apply Normalization (Standard Scaling & Log Transform)
        x = (x - STATS["x_mean"]) / STATS["x_std"]
        y = (y - STATS["y_mean"]) / STATS["y_std"]
        z = (z - STATS["z_mean"]) / STATS["z_std"]
        t = (t - STATS["time_mean"]) / STATS["time_std"]
        c = np.log1p(c)

        # Stack into feature matrix (L, 6)
        features = np.stack([x, y, z, t, c, a], axis=1)

        # ---------------------------------------------------------------------
        # 5. Padding
        # ---------------------------------------------------------------------
        L = features.shape[0]
        if L < SEQ_LEN:
            padding = np.zeros((SEQ_LEN - L, N_FEATURES), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)

        features_tensor = torch.tensor(features, dtype=torch.float32)

        # ---------------------------------------------------------------------
        # 6. Return Targets
        # ---------------------------------------------------------------------
        if self.mode != "test":
            azimuth = row["azimuth"]
            zenith = row["zenith"]
            targets = torch.tensor([azimuth, zenith], dtype=torch.float32)
            return features_tensor, targets
        else:
            # Return dummy targets for test set to maintain consistent signature
            return features_tensor, torch.zeros(2, dtype=torch.float32)
