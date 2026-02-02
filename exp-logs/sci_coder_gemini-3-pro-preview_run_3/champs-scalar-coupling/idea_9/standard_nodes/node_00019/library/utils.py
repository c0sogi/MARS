import os
import sys
import logging
import json
import numpy as np
import pandas as pd
import torch
from library.config import Config


def setup_logger(name="experiment", log_file=None, level=logging.INFO):
    """
    Configures and returns a logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def calculate_log_mae(y_true, y_pred, types):
    """
    Calculates the Log Mean Absolute Error metric.

    Args:
        y_true (np.array or torch.Tensor): True target values.
        y_pred (np.array or torch.Tensor): Predicted values.
        types (np.array or list): Coupling types corresponding to each sample.

    Returns:
        float: The Log MAE score.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(types, list):
        types = np.array(types)

    # Ensure 1D arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    unique_types = np.unique(types)
    log_maes = []

    for t in unique_types:
        mask = types == t
        if np.sum(mask) > 0:
            mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            # Avoid log(0)
            log_mae = np.log(mae + 1e-9) if mae > 1e-9 else -9.0
            log_maes.append(log_mae)

    if not log_maes:
        return 0.0

    return np.mean(log_maes)


class Standardizer:
    """
    Handles per-type standardization of targets using vectorized lookup.
    Cite solution_lesson_node_00017: Canonical Indexing Prevents Silent Data Corruption.
    """

    def __init__(self):
        self.num_types = len(Config.COUPLING_TYPES)
        # Initialize with defaults (mean=0, std=1)
        self.means = torch.zeros(self.num_types)
        self.stds = torch.ones(self.num_types)
        self.type_map = {t: i for i, t in enumerate(Config.COUPLING_TYPES)}
        self.fitted = False

    def fit(self, df, target_col="scalar_coupling_constant", type_col="type"):
        """
        Computes mean and std for each coupling type.
        """
        grouped = df.groupby(type_col)[target_col].agg(["mean", "std"])

        # Fill tensors
        for t_str, row in grouped.iterrows():
            if t_str in self.type_map:
                idx = self.type_map[t_str]
                self.means[idx] = float(row["mean"])
                self.stds[idx] = float(row["std"])

        self.fitted = True
        return self

    def transform(self, values, types):
        """
        Standardizes values based on their type.
        z = (y - mean) / std
        """
        if isinstance(values, torch.Tensor):
            device = values.device
            if self.means.device != device:
                self.means = self.means.to(device)
                self.stds = self.stds.to(device)

            # Vectorized lookup
            m = self.means[types]
            s = self.stds[types]
            return (values - m) / (s + 1e-9)
        else:
            # Numpy fallback
            m = self.means.numpy()[types]
            s = self.stds.numpy()[types]
            return (values - m) / (s + 1e-9)

    def inverse_transform(self, values, types):
        """
        Reverts standardized values to original scale.
        y = z * std + mean
        """
        if isinstance(values, torch.Tensor):
            device = values.device
            if self.means.device != device:
                self.means = self.means.to(device)
                self.stds = self.stds.to(device)

            m = self.means[types]
            s = self.stds[types]
            return values * s + m
        else:
            # Numpy fallback
            m = self.means.numpy()[types]
            s = self.stds.numpy()[types]
            return values * s + m

    def save(self, path):
        """Saves statistics to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stats_dict = {
            "means": self.means.cpu().tolist(),
            "stds": self.stds.cpu().tolist(),
        }
        with open(path, "w") as f:
            json.dump(stats_dict, f, indent=4)

    def load(self, path):
        """Loads statistics from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stats file not found at {path}")
        with open(path, "r") as f:
            stats_dict = json.load(f)

        self.means = torch.tensor(stats_dict["means"])
        self.stds = torch.tensor(stats_dict["stds"])
        self.fitted = True
        return self


def _load_csv_cached(csv_path, cache_name, load_cached_data=True):
    """
    Generic helper to load a CSV with Parquet caching.
    """
    cache_dir = Config.PROCESSED_DATA_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Reloading from source.")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Save cache
    try:
        df.to_parquet(cache_path, index=False)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

    return df


def load_dipole_moments(load_cached_data=True):
    return _load_csv_cached(
        Config.DIPOLE_MOMENTS_PATH, "dipole_moments", load_cached_data
    )


def load_potential_energy(load_cached_data=True):
    return _load_csv_cached(
        Config.POTENTIAL_ENERGY_PATH, "potential_energy", load_cached_data
    )


def load_magnetic_shielding(load_cached_data=True):
    return _load_csv_cached(
        Config.MAGNETIC_SHIELDING_PATH, "magnetic_shielding", load_cached_data
    )


def load_mulliken_charges(load_cached_data=True):
    return _load_csv_cached(
        Config.MULLIKEN_CHARGES_PATH, "mulliken_charges", load_cached_data
    )


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
