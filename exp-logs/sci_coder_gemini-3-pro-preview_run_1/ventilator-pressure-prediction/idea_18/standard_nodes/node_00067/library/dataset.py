import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.features import get_processed_data


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for the Ventilator Pressure Prediction task.
    Wraps pre-processed numpy arrays reshaped into sequences.
    """

    def __init__(
        self,
        X: np.ndarray,
        u_out: np.ndarray,
        y: np.ndarray = None,
        ids: np.ndarray = None,
    ):
        """
        Args:
            X (np.ndarray): Input features of shape (N_breaths, 80, N_features).
            u_out (np.ndarray): Binary control flag of shape (N_breaths, 80).
            y (np.ndarray, optional): Target pressure of shape (N_breaths, 80).
            ids (np.ndarray, optional): Time step IDs of shape (N_breaths, 80).
        """
        self.X = X
        self.u_out = u_out
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing tensors for a single breath.
        """
        data = {
            "input": torch.tensor(self.X[idx], dtype=torch.float32),
            "u_out": torch.tensor(self.u_out[idx], dtype=torch.float32),
        }

        if self.y is not None:
            data["target"] = torch.tensor(self.y[idx], dtype=torch.float32)

        if self.ids is not None:
            data["ids"] = torch.tensor(self.ids[idx], dtype=torch.int64)

        return data


def prepare_data(
    config: Config, split: str, load_cached_data: bool = True
) -> VentilatorDataset:
    """
    Loads data, performs reshaping for time-series format, and wraps it in a Dataset.
    Implements caching of reshaped numpy arrays to speed up subsequent runs.

    Args:
        config (Config): Configuration object.
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        VentilatorDataset: The prepared dataset.
    """
    # Ensure working directory exists
    cache_dir = config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)

    # Define cache filenames
    suffix = "_debug" if config.debug else ""
    path_X = os.path.join(cache_dir, f"{split}_X{suffix}.npy")
    path_uout = os.path.join(cache_dir, f"{split}_uout{suffix}.npy")
    path_y = os.path.join(cache_dir, f"{split}_y{suffix}.npy")
    path_ids = os.path.join(cache_dir, f"{split}_ids{suffix}.npy")

    # Check if we can load from cache
    files_exist = (
        os.path.exists(path_X)
        and os.path.exists(path_uout)
        and os.path.exists(path_ids)
    )
    if split != "test":
        files_exist = files_exist and os.path.exists(path_y)

    if load_cached_data and files_exist:
        print(f"Loading cached tensor data for {split} from {cache_dir}...")
        X = np.load(path_X)
        u_out = np.load(path_uout)
        ids = np.load(path_ids)
        y = np.load(path_y) if split != "test" else None
        return VentilatorDataset(X, u_out, y, ids)

    # If cache miss or force reload, process from scratch
    print(f"Processing tensor data for {split}...")

    # 1. Get flat feature-engineered DataFrame
    # This function handles its own caching of the flat parquet file
    df = get_processed_data(config, split, load_cached_data=load_cached_data)

    # 2. Sort to ensure correct sequence order
    df = df.sort_values([config.BREATH_COL, "time_step"])

    # 3. Determine shapes
    n_breaths = df[config.BREATH_COL].nunique()
    steps_per_breath = 80  # Fixed for this dataset

    # Validation check for data integrity
    if len(df) != n_breaths * steps_per_breath:
        raise ValueError(
            f"Data length ({len(df)}) is not a multiple of steps_per_breath ({steps_per_breath}). "
            "Check if data is complete or if debug subsampling broke integrity."
        )

    # 4. Extract and Reshape Features
    # Input features = Continuous Scaled + Binary (Unscaled)
    # Note: u_out is included in inputs (for model context) AND separated (for loss masking)
    feature_cols = config.CONT_FEATURES + config.BINARY_FEATURES

    # Extract numpy arrays
    X_flat = df[feature_cols].values.astype(np.float32)
    u_out_flat = df["u_out"].values.astype(np.float32)
    ids_flat = df[config.ID_COL].values.astype(np.int64)

    # Reshape to (N, 80, Features)
    X = X_flat.reshape(n_breaths, steps_per_breath, -1)
    u_out = u_out_flat.reshape(n_breaths, steps_per_breath)
    ids = ids_flat.reshape(n_breaths, steps_per_breath)

    # Handle Target
    y = None
    if split != "test":
        y_flat = df[config.TARGET_COL].values.astype(np.float32)
        y = y_flat.reshape(n_breaths, steps_per_breath)
        np.save(path_y, y)

    # 5. Save to Cache
    print(f"Saving reshaped tensors to {cache_dir}...")
    np.save(path_X, X)
    np.save(path_uout, u_out)
    np.save(path_ids, ids)

    return VentilatorDataset(X, u_out, y, ids)
