import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import RobustScaler
from library.config import Config
from library.utils import generate_cache_hash, ensure_dir
from library.features import get_processed_dataset


class VentilatorDataset(Dataset):
    """
    PyTorch Dataset for Ventilator Pressure Prediction.
    Holds pre-processed tensors for features, targets, masks, and theoretical baselines.
    """

    def __init__(
        self,
        X: np.ndarray,
        u_out: np.ndarray,
        p_theory: np.ndarray,
        y: np.ndarray = None,
        ids: np.ndarray = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.u_out = torch.tensor(u_out, dtype=torch.float32)
        self.p_theory = torch.tensor(p_theory, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.ids = ids  # Keep IDs for submission mapping if needed

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {
            "x": self.X[idx],
            "u_out": self.u_out[idx],
            "p_theory": self.p_theory[idx],
        }
        if self.y is not None:
            item["y"] = self.y[idx]
        if self.ids is not None:
            item["ids"] = self.ids[idx]
        return item


def _get_scaler_paths():
    """Returns paths for scaler attributes."""
    center_path = os.path.join(Config.cache_dir, "scaler_center.npy")
    scale_path = os.path.join(Config.cache_dir, "scaler_scale.npy")
    return center_path, scale_path


def _save_scaler(scaler: RobustScaler):
    """Saves RobustScaler attributes to npy files (avoiding pickle)."""
    center_path, scale_path = _get_scaler_paths()
    ensure_dir(center_path)
    np.save(center_path, scaler.center_)
    np.save(scale_path, scaler.scale_)


def _load_scaler() -> RobustScaler:
    """Loads RobustScaler from npy files."""
    center_path, scale_path = _get_scaler_paths()
    if not os.path.exists(center_path) or not os.path.exists(scale_path):
        raise FileNotFoundError(
            "Scaler parameters not found. Train data must be processed first."
        )

    scaler = RobustScaler()
    scaler.center_ = np.load(center_path)
    scaler.scale_ = np.load(scale_path)
    return scaler


def prepare_data(split: str, load_cached_data: bool = True) -> VentilatorDataset:
    """
    Loads, preprocesses, scales, and reshapes data for the given split.
    Implements strict caching logic using numpy arrays.

    Args:
        split (str): 'train', 'val', or 'test'.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        VentilatorDataset: The ready-to-use PyTorch dataset.
    """
    # 1. Generate Cache Hash
    # Hash depends on split, debug mode, features, and sequence length
    config_dict = {
        "split": split,
        "debug": Config.debug,
        "feature_cols": Config.feature_cols,
        "seq_len": Config.seq_len,
        "logic_version": "v2_numpy_cache",
    }
    cache_hash = generate_cache_hash(config_dict)

    # Define cache file paths
    cache_prefix = os.path.join(Config.cache_dir, f"{split}_{cache_hash}")
    path_X = f"{cache_prefix}_X.npy"
    path_uout = f"{cache_prefix}_uout.npy"
    path_theory = f"{cache_prefix}_theory.npy"
    path_y = f"{cache_prefix}_y.npy"
    path_ids = f"{cache_prefix}_ids.npy"

    # 2. Try Loading from Cache
    if load_cached_data:
        files_exist = (
            os.path.exists(path_X)
            and os.path.exists(path_uout)
            and os.path.exists(path_theory)
            and (split == "test" or os.path.exists(path_y))
        )

        if files_exist:
            print(f"Loading cached {split} tensors from {Config.cache_dir}...")
            X = np.load(path_X)
            u_out = np.load(path_uout)
            p_theory = np.load(path_theory)
            ids = np.load(path_ids) if os.path.exists(path_ids) else None
            y = np.load(path_y) if split != "test" else None

            return VentilatorDataset(X, u_out, p_theory, y, ids)
        else:
            print(
                f"Cache miss for {split} (hash: {cache_hash}). Processing from scratch..."
            )

    # 3. Process from Scratch
    # Load dataframe with physics features
    df = get_processed_dataset(split, load_cached_data=load_cached_data)

    # Ensure sorting
    df = df.sort_values(by=[Config.breath_id_col, Config.time_col])

    # Extract raw arrays
    # Note: 'theoretical_pressure' is created by add_physics_features in library.features
    if "theoretical_pressure" not in df.columns:
        raise ValueError(
            "Column 'theoretical_pressure' missing. Check feature engineering."
        )

    feature_data = df[Config.feature_cols].values.astype(np.float32)
    u_out_data = df["u_out"].values.astype(np.float32)
    theory_data = df["theoretical_pressure"].values.astype(np.float32)
    ids_data = df[Config.id_col].values.astype(np.int64)

    y_data = None
    if split != "test":
        y_data = df[Config.target_col].values.astype(np.float32)

    # 4. Scaling (RobustScaler)
    # We scale the feature_data. u_out and theory are kept as is (or handled separately).
    if split == "train":
        print("Fitting RobustScaler on training data...")
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        feature_data = scaler.fit_transform(feature_data)
        _save_scaler(scaler)
    else:
        print(f"Transforming {split} data using saved scaler...")
        scaler = _load_scaler()
        feature_data = scaler.transform(feature_data)

    # 5. Reshape to Sequences (N_breaths, 80, Features)
    # Verify integrity
    num_rows = len(df)
    if num_rows % Config.seq_len != 0:
        raise ValueError(
            f"Dataset length {num_rows} is not divisible by seq_len {Config.seq_len}."
        )

    num_breaths = num_rows // Config.seq_len

    X = feature_data.reshape(num_breaths, Config.seq_len, -1)
    u_out = u_out_data.reshape(num_breaths, Config.seq_len)
    p_theory = theory_data.reshape(num_breaths, Config.seq_len)
    ids = ids_data.reshape(num_breaths, Config.seq_len)

    y = None
    if y_data is not None:
        y = y_data.reshape(num_breaths, Config.seq_len)

    # 6. Save to Cache
    print(f"Saving processed {split} tensors to cache...")
    ensure_dir(path_X)
    np.save(path_X, X)
    np.save(path_uout, u_out)
    np.save(path_theory, p_theory)
    np.save(path_ids, ids)
    if y is not None:
        np.save(path_y, y)

    return VentilatorDataset(X, u_out, p_theory, y, ids)
