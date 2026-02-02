import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import load_sensor_geometry, angles_to_vector


class NeutrinoDataset(Dataset):
    """
    PyTorch Dataset for Neutrino Direction Prediction.
    Reads data from Parquet batch files using a metadata index.
    """

    def __init__(self, metadata, geometry, mode="train"):
        """
        Args:
            metadata (pd.DataFrame): Metadata containing event_id, file_path, indices, etc.
            geometry (pd.DataFrame): Sensor geometry (index=sensor_id, cols=[x, y, z]).
            mode (str): 'train', 'val', or 'test'.
        """
        self.metadata = metadata.reset_index(drop=True)
        self.mode = mode

        # Convert geometry to numpy array for fast O(1) lookup
        # We assume sensor_ids are integers. We create an array large enough to hold the max ID.
        max_id = int(geometry.index.max())
        self.geo_array = np.zeros((max_id + 1, 3), dtype=np.float32)
        # Fill the array with geometry data
        # geometry.index is sensor_id
        self.geo_array[geometry.index] = geometry[["x", "y", "z"]].values

        # LRU Cache for batch dataframes
        # Each worker process will have its own cache
        self.batch_cache = {}
        self.cache_order = []
        self.cache_size = (
            10  # Keep 10 batch files in memory per worker (~1.7GB per worker)
        )

    def _get_batch_df(self, rel_path):
        """
        Retrieves a batch dataframe from cache or loads it from disk.
        """
        if rel_path in self.batch_cache:
            # Move to end (mark as recently used)
            self.cache_order.remove(rel_path)
            self.cache_order.append(rel_path)
            return self.batch_cache[rel_path]

        # Load file
        full_path = os.path.join(Config.INPUT_DIR, rel_path)
        # We only need specific columns to save memory/time
        cols = ["time", "sensor_id", "charge", "auxiliary"]

        # Check if file exists (sanity check)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Batch file not found: {full_path}")

        df = pd.read_parquet(full_path, columns=cols)

        # Manage cache size
        if len(self.cache_order) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.batch_cache[oldest]

        self.batch_cache[rel_path] = df
        self.cache_order.append(rel_path)
        return df

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 1. Load Batch Data
        file_path = row["file_path"]
        first_idx = int(row["first_pulse_index"])
        last_idx = int(row["last_pulse_index"])

        df_batch = self._get_batch_df(file_path)

        # Slice the specific event
        # iloc is inclusive-exclusive, but last_pulse_index is the index of the last row
        event_pulses = df_batch.iloc[first_idx : last_idx + 1]

        # 2. Filter Auxiliary (Noise)
        mask = ~event_pulses["auxiliary"]
        event_pulses = event_pulses[mask]

        # Handle case where all pulses are auxiliary
        if len(event_pulses) == 0:
            # Return zero-filled tensor
            x = torch.zeros((Config.SEQ_LEN, Config.NUM_FEATURES), dtype=torch.float32)
            if self.mode == "test":
                return x, torch.tensor(row["event_id"], dtype=torch.long)
            else:
                # Dummy target (z-axis unit vector)
                return x, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

        # 3. Sort by time
        # Extract values for processing
        times = event_pulses["time"].values
        charges = event_pulses["charge"].values
        sensor_ids = event_pulses["sensor_id"].values.astype(int)

        # Sort indices
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        charges = charges[sort_idx]
        sensor_ids = sensor_ids[sort_idx]

        # 4. Feature Engineering

        # Geometry lookup: Get x, y, z for each sensor
        # Shape (L, 3)
        coords = self.geo_array[sensor_ids]

        # Normalization
        # Time: Standardize
        norm_time = (times - Config.NORM_TIME_MEAN) / Config.NORM_TIME_STD

        # Coordinates: Standardize
        norm_coords = (coords - Config.NORM_COORD_MEAN) / Config.NORM_COORD_STD

        # Charge: Log1p
        norm_charge = np.log1p(charges)

        # Stack features: x, y, z, time, charge
        # shapes: (L, 3), (L,), (L,) -> (L, 5)
        features = np.column_stack([norm_coords, norm_time, norm_charge]).astype(
            np.float32
        )

        # 5. Pad or Truncate to SEQ_LEN
        L = features.shape[0]
        target_len = Config.SEQ_LEN

        final_tensor = np.zeros((target_len, Config.NUM_FEATURES), dtype=np.float32)

        if L >= target_len:
            # Truncate: take first SEQ_LEN pulses (earliest in time)
            final_tensor[:] = features[:target_len]
        else:
            # Pad: fill first L rows, rest remain zeros
            final_tensor[:L] = features

        x = torch.from_numpy(final_tensor)

        # 6. Return Data
        if self.mode == "test":
            return x, torch.tensor(row["event_id"], dtype=torch.long)
        else:
            # Get target
            az = row["azimuth"]
            zen = row["zenith"]
            # Convert to 3D unit vector
            vec = angles_to_vector(az, zen)
            y = torch.from_numpy(vec).float()
            return x, y


def get_dataloaders():
    """
    Creates DataLoaders for train and validation sets.

    Returns:
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.
    """
    # 1. Load Metadata
    train_meta = pd.read_parquet(Config.TRAIN_META_PATH)
    val_meta = pd.read_parquet(Config.VAL_META_PATH)

    # 2. Debug Subsetting
    if Config.DEBUG:
        print(
            f"DEBUG Mode: Sampling {Config.DEBUG_SUBSET_SIZE} events for training/validation..."
        )
        train_meta = train_meta.iloc[: Config.DEBUG_SUBSET_SIZE]
        val_meta = val_meta.iloc[: Config.DEBUG_SUBSET_SIZE]

    # 3. Load Geometry
    # Using load_cached_data=True to use the cached parquet file if available
    geometry = load_sensor_geometry(load_cached_data=True)

    # 4. Create Datasets
    train_dataset = NeutrinoDataset(train_meta, geometry, mode="train")
    val_dataset = NeutrinoDataset(val_meta, geometry, mode="val")

    # 5. Create DataLoaders
    # persistent_workers=True keeps the workers (and their batch_cache) alive between epochs
    # Only enable persistent_workers if num_workers > 0
    use_persistent_workers = Config.NUM_WORKERS > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        persistent_workers=use_persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        persistent_workers=use_persistent_workers,
    )

    return train_loader, val_loader


def get_test_dataloader():
    """
    Creates DataLoader for the test set.

    Returns:
        test_loader (DataLoader): Loader for test data.
    """
    # 1. Load Metadata
    test_meta = pd.read_parquet(Config.TEST_META_PATH)

    # 1.5 Debug Subsetting
    if Config.DEBUG:
        print(f"DEBUG Mode: Sampling {Config.DEBUG_SUBSET_SIZE} events for testing...")
        test_meta = test_meta.iloc[: Config.DEBUG_SUBSET_SIZE]

    # 2. Load Geometry
    geometry = load_sensor_geometry(load_cached_data=True)

    # 3. Create Dataset
    test_dataset = NeutrinoDataset(test_meta, geometry, mode="test")

    # 4. Create DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False,
        persistent_workers=(Config.NUM_WORKERS > 0),
    )

    return test_loader
