import os
import logging
import numpy as np
import pandas as pd
import torch
from library.config import Config


def setup_logger(name="logger", log_file=None, level=logging.INFO):
    """
    Sets up a logger with console and optional file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already exists
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def calculate_log_mae(y_true, y_pred, types):
    """
    Calculates the Log of Mean Absolute Error for each type, averaged across types.
    Metric = Mean( Log( MAE(type) ) )

    Args:
        y_true: array-like of shape (N,)
        y_pred: array-like of shape (N,)
        types: array-like of shape (N,) indicating coupling types

    Returns:
        float: The metric score
    """
    # Convert torch tensors to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(types, torch.Tensor):
        types = types.detach().cpu().numpy()

    # Flatten arrays
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    types = np.array(types).flatten()

    unique_types = np.unique(types)
    log_maes = []

    for t in unique_types:
        mask = types == t
        if np.sum(mask) > 0:
            mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            # Use a small epsilon to avoid log(0) if perfect prediction
            log_mae = np.log(mae + 1e-9)
            log_maes.append(log_mae)

    if not log_maes:
        return 0.0

    return np.mean(log_maes)


class GroupStandardizer:
    """
    Standardizes target variables independently for each coupling type.
    z = (y - mean_t) / std_t
    """

    def __init__(self):
        self.means = {}
        self.stds = {}
        self.stats_file = Config.STATS_CACHE

    def fit(
        self,
        df,
        target_col="scalar_coupling_constant",
        type_col="type",
        load_cached_data=True,
    ):
        """
        Computes mean and std for each type from the dataframe.
        Supports caching to disk using structured numpy arrays (no pickle).
        """
        # Ensure working directory exists
        os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)

        # 1. Try to load from cache
        if load_cached_data and os.path.exists(self.stats_file):
            try:
                # Load structured array
                # Using allow_pickle=False ensures we are using pure numpy formats
                data = np.load(self.stats_file, allow_pickle=False)

                # Reconstruct dicts
                self.means = {}
                self.stds = {}
                for row in data:
                    t = str(row["type"])
                    self.means[t] = float(row["mean"])
                    self.stds[t] = float(row["std"])

                print(f"Loaded standardizer stats from {self.stats_file}")
                return
            except Exception as e:
                print(f"Failed to load standardizer cache: {e}. Recomputing...")

        # 2. Compute from scratch
        if df is None:
            raise ValueError("DataFrame is None and cache could not be loaded.")

        print("Computing standardizer stats...")
        groups = df.groupby(type_col)[target_col]
        means = groups.mean()
        stds = groups.std()

        self.means = means.to_dict()
        self.stds = stds.to_dict()

        # Handle zero std (unlikely but safe to handle)
        for k, v in self.stds.items():
            if v == 0:
                self.stds[k] = 1.0

        # 3. Save to cache using structured array to avoid pickle
        # Define dtype: Type string length max 10, mean float, std float
        dtype = [("type", "U10"), ("mean", "f8"), ("std", "f8")]
        data_list = []
        for t in self.means.keys():
            data_list.append((t, self.means[t], self.stds[t]))

        structured_data = np.array(data_list, dtype=dtype)
        print(f"Saving standardizer stats to {self.stats_file}")
        np.save(self.stats_file, structured_data)

    def transform(self, values, types):
        """
        Standardizes values: (value - mean) / std
        """
        values = np.array(values)
        types = np.array(types)
        result = np.zeros_like(values, dtype=np.float32)

        for t, mean in self.means.items():
            if t in self.stds:
                std = self.stds[t]
                mask = types == t
                if np.any(mask):
                    result[mask] = (values[mask] - mean) / std
        return result

    def inverse_transform(self, values, types):
        """
        Inverse standardizes values: value * std + mean
        """
        values = np.array(values)
        types = np.array(types)
        result = np.zeros_like(values, dtype=np.float32)

        for t, mean in self.means.items():
            if t in self.stds:
                std = self.stds[t]
                mask = types == t
                if np.any(mask):
                    result[mask] = (values[mask] * std) + mean
        return result
