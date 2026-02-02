import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_sensor_geometry


class IceCubeBatchDataset(Dataset):
    """
    Loads a single batch of IceCube data into memory.
    Handles preprocessing and caching of tensors to .npy files to minimize I/O overhead.
    """

    def __init__(
        self, batch_id, meta_df, sensor_geo, mode="train", load_cached_data=True
    ):
        """
        Args:
            batch_id (int): The ID of the batch to load.
            meta_df (pd.DataFrame): Metadata containing event indices and targets.
            sensor_geo (pd.DataFrame): Sensor geometry data.
            mode (str): 'train' (returns X, y) or 'test' (returns X).
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        self.batch_id = batch_id
        self.mode = mode
        self.cache_dir = os.path.join(Config.WORKING_DIR, "cache")

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

        # Define cache file paths
        self.cache_X = os.path.join(self.cache_dir, f"{mode}_batch_{batch_id}_X.npy")
        self.cache_y = os.path.join(self.cache_dir, f"{mode}_batch_{batch_id}_y.npy")
        self.cache_ids = os.path.join(
            self.cache_dir, f"{mode}_batch_{batch_id}_ids.npy"
        )

        # Filter metadata for this batch
        self.batch_meta = meta_df[meta_df["batch_id"] == batch_id].copy()

        # Initialize empty if no events found (edge case)
        if self.batch_meta.empty:
            self.X = torch.empty(0, Config.NUM_PULSES, Config.INPUT_DIM)
            self.ids = torch.empty(0, dtype=torch.long)
            self.y = torch.empty(0, Config.OUTPUT_DIM) if mode != "test" else None
            return

        # Attempt to load from cache
        if load_cached_data and self._check_cache_exists():
            self._load_cache()
        else:
            self._process_and_cache(sensor_geo)

    def _check_cache_exists(self):
        """Checks if all required cache files exist."""
        if not (os.path.exists(self.cache_X) and os.path.exists(self.cache_ids)):
            return False
        if self.mode != "test" and not os.path.exists(self.cache_y):
            return False
        return True

    def _load_cache(self):
        """Loads data from .npy files and converts to tensors."""
        try:
            X_np = np.load(self.cache_X)
            ids_np = np.load(self.cache_ids)

            self.X = torch.tensor(X_np, dtype=torch.float32)
            self.ids = torch.tensor(ids_np, dtype=torch.long)

            if self.mode != "test":
                y_np = np.load(self.cache_y)
                self.y = torch.tensor(y_np, dtype=torch.float32)
            else:
                self.y = None
        except Exception:
            # If loading fails (e.g. corrupt file), force re-processing could be an option,
            # but here we just raise or let it fail.
            raise RuntimeError(f"Failed to load cache for batch {self.batch_id}")

    def _process_and_cache(self, sensor_geo):
        """Reads parquet, processes features, samples pulses, and saves to cache."""
        # 1. Load Raw Batch Data
        batch_file = os.path.join(
            Config.INPUT_DIR, self.mode, f"batch_{self.batch_id}.parquet"
        )
        if not os.path.exists(batch_file):
            raise FileNotFoundError(f"Batch file not found: {batch_file}")

        df_batch = pd.read_parquet(batch_file)

        # 2. Merge Sensor Geometry
        # sensor_geo index should be sensor_id
        df_batch = df_batch.join(sensor_geo, on="sensor_id", how="left")

        # 3. Feature Engineering (Vectorized)
        # Cast auxiliary to float
        df_batch["auxiliary"] = df_batch["auxiliary"].astype(np.float32)

        # Log transform charge: log10(max(charge, 1e-3))
        df_batch["charge"] = np.log10(np.clip(df_batch["charge"], 1e-3, None)).astype(
            np.float32
        )

        # Scale coordinates
        df_batch["x"] /= Config.COORD_SCALE
        df_batch["y"] /= Config.COORD_SCALE
        df_batch["z"] /= Config.COORD_SCALE

        # Extract columns as numpy array for fast slicing
        # Feature Order: x, y, z, time, charge, auxiliary
        feature_cols = ["x", "y", "z", "time", "charge", "auxiliary"]
        data_arr = df_batch[feature_cols].to_numpy(dtype=np.float32)

        # 4. Construct Event Tensors
        num_events = len(self.batch_meta)
        X_np = np.zeros(
            (num_events, Config.NUM_PULSES, Config.INPUT_DIM), dtype=np.float32
        )

        # Metadata indices
        starts = self.batch_meta["first_pulse_index"].values
        ends = self.batch_meta["last_pulse_index"].values

        # Column indices for specific features
        t_col_idx = 3
        charge_col_idx = 4

        # Iterate over events to normalize time and sample pulses
        for i in range(num_events):
            s, e = starts[i], ends[i]
            # Slice pulses for this event (inclusive of e based on description "index of last row")
            event_pulses = data_arr[s : e + 1]

            if len(event_pulses) == 0:
                continue

            # Time Normalization: relative to min time in event
            t_min = np.min(event_pulses[:, t_col_idx])
            # Create a copy to avoid modifying the shared array for subsequent logic if needed
            # (though here we just write to X_np)
            current_pulses = event_pulses.copy()
            current_pulses[:, t_col_idx] = (
                current_pulses[:, t_col_idx] - t_min
            ) / Config.TIME_SCALE

            # Sampling Strategy: Top N by charge
            n_pulses = current_pulses.shape[0]

            if n_pulses > Config.NUM_PULSES:
                # argsort returns indices that sort the array
                sort_idx = np.argsort(current_pulses[:, charge_col_idx])
                # Take the indices of the highest charges (end of the sorted array)
                top_idx = sort_idx[-Config.NUM_PULSES :]
                selected = current_pulses[top_idx]
                X_np[i, :, :] = selected
            else:
                # Pad with zeros (X_np is already zeros)
                X_np[i, :n_pulses, :] = current_pulses

        # 5. Prepare Targets and IDs
        ids_np = self.batch_meta["event_id"].values.astype(np.int64)

        if self.mode != "test":
            az = self.batch_meta["azimuth"].values.astype(np.float32)
            ze = self.batch_meta["zenith"].values.astype(np.float32)
            y_np = np.stack([az, ze], axis=1)
        else:
            y_np = None

        # 6. Save to Cache
        np.save(self.cache_X, X_np)
        np.save(self.cache_ids, ids_np)
        if y_np is not None:
            np.save(self.cache_y, y_np)

        # 7. Set Attributes
        self.X = torch.tensor(X_np, dtype=torch.float32)
        self.ids = torch.tensor(ids_np, dtype=torch.long)
        if y_np is not None:
            self.y = torch.tensor(y_np, dtype=torch.float32)
        else:
            self.y = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Returns:
            Train/Val: (X_tensor, y_tensor)
            Test: (X_tensor,)
        """
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return (self.X[idx],)
