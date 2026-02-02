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
    Handles per-type standardization of targets.
    """

    def __init__(self):
        self.stats = {}  # {type: {'mean': float, 'std': float}}

    def fit(self, df, target_col="scalar_coupling_constant", type_col="type"):
        """
        Computes mean and std for each coupling type.
        """
        self.stats = {}
        grouped = df.groupby(type_col)[target_col].agg(["mean", "std"])

        for t, row in grouped.iterrows():
            self.stats[str(t)] = {"mean": float(row["mean"]), "std": float(row["std"])}
        return self

    def transform(self, values, types):
        """
        Standardizes values based on their type.
        z = (y - mean) / std
        """
        if isinstance(values, torch.Tensor):
            device = values.device
            is_tensor = True
            values = values.detach().cpu().numpy()
        else:
            is_tensor = False

        result = np.zeros_like(values, dtype=np.float32)

        # We assume types corresponds to values index-wise
        # Optimize by processing masks for each known type
        for t, stat in self.stats.items():
            mask = types == t
            if np.any(mask):
                result[mask] = (values[mask] - stat["mean"]) / (stat["std"] + 1e-9)

        if is_tensor:
            return torch.tensor(result, dtype=torch.float32, device=device)
        return result

    def inverse_transform(self, values, types):
        """
        Reverts standardized values to original scale.
        y = z * std + mean
        """
        if isinstance(values, torch.Tensor):
            device = values.device
            is_tensor = True
            values = values.detach().cpu().numpy()
        else:
            is_tensor = False

        result = np.zeros_like(values, dtype=np.float32)

        for t, stat in self.stats.items():
            mask = types == t
            if np.any(mask):
                result[mask] = values[mask] * stat["std"] + stat["mean"]

        if is_tensor:
            return torch.tensor(result, dtype=torch.float32, device=device)
        return result

    def save(self, path):
        """Saves statistics to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=4)

    def load(self, path):
        """Loads statistics from a JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stats file not found at {path}")
        with open(path, "r") as f:
            self.stats = json.load(f)
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
