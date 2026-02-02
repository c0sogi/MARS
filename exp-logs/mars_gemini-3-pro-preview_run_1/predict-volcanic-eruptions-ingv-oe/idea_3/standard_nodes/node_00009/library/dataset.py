import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from library.config import Config


def load_raw_segment(file_path):
    """
    Reads a sensor data CSV file, fills missing values, and returns the raw numpy array.

    Args:
        file_path (str): Full path to the CSV file.

    Returns:
        np.ndarray: Processed sensor data of shape (SEQ_LEN, NUM_SENSORS).
    """
    try:
        # Load data with specific dtype to save memory
        df = pd.read_csv(file_path, dtype="float32")

        # Fill NaNs:
        # 1. Fill with column mean (best guess for that sensor in that segment)
        # 2. Fill remaining (if column is all NaN) with 0.0
        df = df.fillna(df.mean()).fillna(0.0)

        # Ensure correct column order and existence based on Config
        sensors = Config.SENSORS
        # Create missing columns if necessary (though unlikely given analysis)
        for sensor in sensors:
            if sensor not in df.columns:
                df[sensor] = 0.0

        # Return as float32 numpy array
        return df[sensors].values.astype(np.float32)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        # Return zeros of correct shape to prevent pipeline crash
        return np.zeros((Config.SEQ_LEN, Config.NUM_SENSORS), dtype=np.float32)


class SeismicDataset(Dataset):
    """
    PyTorch Dataset for Seismic Eruption Prediction.
    Performs on-the-fly Robust Scaling and reshaping for 1D-ResNet.
    """

    def __init__(self, data, targets=None):
        """
        Args:
            data (np.ndarray): Raw sensor data of shape (N, SEQ_LEN, NUM_SENSORS).
            targets (np.ndarray, optional): Target values of shape (N,).
        """
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retrieve raw sample: (SEQ_LEN, NUM_SENSORS) -> (60001, 10)
        x = self.data[idx]

        # --- On-the-fly Preprocessing ---

        # 1. Robust Scaling: (x - median) / IQR
        # We compute statistics per sensor (axis 0 of the sensor dimension,
        # but here x is (Time, Sensors), so we compute over Time (axis 0))
        median = np.median(x, axis=0)
        q25 = np.quantile(x, 0.25, axis=0)
        q75 = np.quantile(x, 0.75, axis=0)
        iqr = q75 - q25

        # Avoid division by zero for constant signals
        iqr = np.where(iqr == 0, 1.0, iqr)

        # Apply scaling
        x_scaled = (x - median) / (iqr + 1e-6)

        # 2. Reshape for 1D-ResNet: (Channels, Time) -> (10, 60001)
        # Current shape is (60001, 10), so we transpose
        x_scaled = x_scaled.transpose(1, 0)

        # Convert to Tensor
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        # Return tuple if targets exist, else just input
        if self.targets is not None:
            y_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
            return x_tensor, y_tensor

        return x_tensor


def get_dataset(metadata_path, split_name, load_cached_data=True):
    """
    Factory function to load data (from cache or raw CSVs) and return a SeismicDataset.
    Implements the required caching mechanism.

    Args:
        metadata_path (str): Path to metadata CSV.
        split_name (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        SeismicDataset: The ready-to-use dataset.
    """
    # Ensure working directory exists
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    data_cache_path = os.path.join(cache_dir, f"{split_name}_raw_data.npy")
    target_cache_path = os.path.join(cache_dir, f"{split_name}_targets.npy")

    # 1. Try Loading from Cache
    if load_cached_data and os.path.exists(data_cache_path):
        # Determine if we need targets
        targets_exist = os.path.exists(target_cache_path)

        # If it's test, we don't need targets. If it's train/val, we expect targets.
        if split_name == "test" or targets_exist:
            print(f"Loading cached raw data for {split_name} from {data_cache_path}")
            data = np.load(data_cache_path)

            targets = None
            if targets_exist and split_name != "test":
                targets = np.load(target_cache_path)

            return SeismicDataset(data, targets)

    # 2. Load from Raw CSVs (Cache Miss or Force Reload)
    print(f"Processing raw data for {split_name} from CSVs...")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    df_meta = pd.read_csv(metadata_path)

    # Construct full file paths
    # Metadata contains relative paths e.g., "train/123.csv"
    file_paths = [os.path.join(Config.INPUT_DIR, p) for p in df_meta["file_path"]]

    # Parallel Load
    # Use all available cores to speed up CSV reading
    data_list = Parallel(n_jobs=-1, verbose=0)(
        delayed(load_raw_segment)(fp) for fp in file_paths
    )

    # Stack into a single array: (N, 60001, 10)
    data = np.stack(data_list)

    # Handle Targets
    targets = None
    # We only load targets if it's not the test set and the column exists
    if split_name != "test" and "time_to_eruption" in df_meta.columns:
        targets = df_meta["time_to_eruption"].values.astype(np.float32)

    # 3. Save to Cache
    print(f"Saving raw data cache to {data_cache_path}")
    np.save(data_cache_path, data)
    if targets is not None:
        np.save(target_cache_path, targets)

    return SeismicDataset(data, targets)
