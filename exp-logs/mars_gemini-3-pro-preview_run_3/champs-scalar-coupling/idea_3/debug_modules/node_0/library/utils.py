import os
import torch
import numpy as np
import pandas as pd
from library.config import Config


def calculate_log_mae(y_true, y_pred, types):
    """
    Calculates the Log of the Mean Absolute Error for each scalar coupling type,
    and then averages these log MAEs across types.

    Args:
        y_true (torch.Tensor or np.ndarray): True target values.
        y_pred (torch.Tensor or np.ndarray): Predicted target values.
        types (torch.Tensor or np.ndarray): Coupling type identifiers (integers or strings).

    Returns:
        float: The average Log MAE.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(types, torch.Tensor):
        types = types.detach().cpu().numpy()

    # Flatten arrays to ensure 1D
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    types = types.flatten()

    unique_types = np.unique(types)
    log_maes = []

    for t in unique_types:
        # Create mask for current type
        mask = types == t

        if np.sum(mask) == 0:
            continue

        # Calculate MAE for this type
        diff = np.abs(y_true[mask] - y_pred[mask])
        mae = np.mean(diff)

        # Calculate Log MAE (add small epsilon to avoid log(0))
        log_mae = np.log(mae + 1e-9)
        log_maes.append(log_mae)

    if len(log_maes) == 0:
        return 0.0

    return np.mean(log_maes)


class TypeSpecificStandardizer:
    """
    Handles standardization (z-score normalization) of target variables independently
    for each coupling type. Supports caching of statistics.
    """

    def __init__(self, device=None):
        self.device = device if device else Config.get_device()
        self.means = None
        self.stds = None
        self.num_types = Config.NUM_COUPLING_TYPES
        self.stats_path = Config.STATS_PATH

    def fit(self, df, load_cached_data=True):
        """
        Computes mean and std for each coupling type from the dataframe or loads from cache.

        Args:
            df (pd.DataFrame): Dataframe containing 'type' and 'scalar_coupling_constant'.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        # 1. IF load_cached_data is True: Try to load the file.
        if load_cached_data and os.path.exists(self.stats_path):
            try:
                stats = np.load(self.stats_path)
                self._set_stats(stats)
                return
            except Exception as e:
                print(f"Failed to load cached stats: {e}. Recomputing...")

        # 2. IF loading fails OR load_cached_data is False: Compute from scratch.
        stats = np.zeros((self.num_types, 2))

        # Ensure we cover all types defined in Config
        for t_name, t_idx in Config.COUPLING_TYPE_MAP.items():
            subset = df[df["type"] == t_name]
            if not subset.empty:
                vals = subset["scalar_coupling_constant"].values
                stats[t_idx, 0] = np.mean(vals)
                stats[t_idx, 1] = np.std(vals)
            else:
                # Default to mean=0, std=1 if type is missing (safety fallback)
                stats[t_idx, 0] = 0.0
                stats[t_idx, 1] = 1.0

        # Save the result to the cache directory
        os.makedirs(os.path.dirname(self.stats_path), exist_ok=True)
        np.save(self.stats_path, stats)

        self._set_stats(stats)

    def _set_stats(self, stats):
        """Internal helper to set stats tensors."""
        self.means = torch.tensor(stats[:, 0], device=self.device, dtype=torch.float32)
        self.stds = torch.tensor(stats[:, 1], device=self.device, dtype=torch.float32)

    def _ensure_loaded(self):
        """Ensures stats are loaded (e.g., during inference)."""
        if self.means is None:
            if os.path.exists(self.stats_path):
                stats = np.load(self.stats_path)
                self._set_stats(stats)
            else:
                raise RuntimeError("Standardizer is not fit and no cache file found.")

    def transform(self, values, types):
        """
        Standardizes values based on their type.
        z = (y - mean) / std

        Args:
            values (torch.Tensor): Target values.
            types (torch.Tensor): Integer encoded coupling types.

        Returns:
            torch.Tensor: Standardized values.
        """
        self._ensure_loaded()

        # Gather stats for the specific types in the batch
        batch_means = self.means[types]
        batch_stds = self.stds[types]

        # Handle dimensions if values is (N, 1) or (N,)
        if values.dim() > 1 and batch_means.dim() == 1:
            batch_means = batch_means.view(-1, 1)
            batch_stds = batch_stds.view(-1, 1)

        return (values - batch_means) / (batch_stds + 1e-9)

    def inverse_transform(self, values, types):
        """
        Reverses standardization based on their type.
        y = z * std + mean

        Args:
            values (torch.Tensor): Standardized values.
            types (torch.Tensor): Integer encoded coupling types.

        Returns:
            torch.Tensor: Original scale values.
        """
        self._ensure_loaded()

        batch_means = self.means[types]
        batch_stds = self.stds[types]

        if values.dim() > 1 and batch_means.dim() == 1:
            batch_means = batch_means.view(-1, 1)
            batch_stds = batch_stds.view(-1, 1)

        return values * batch_stds + batch_means
