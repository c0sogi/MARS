import os
import numpy as np
import pandas as pd
import torch
from library.config import Config

# Ensure reproducibility
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)


class GroupScaler:
    """
    Standardizes a target variable based on categorical groups (e.g., coupling types).
    Calculates and stores mean and std for each group to perform per-group Z-score normalization.
    This is crucial for handling the different scales of coupling constants across types (e.g., 1JHC vs 2JHH).
    """

    def __init__(self):
        self.means = {}
        self.stds = {}
        self.fitted = False

    def fit(self, df, target_col, group_col):
        """
        Computes mean and std for each group in the dataframe.

        Args:
            df (pd.DataFrame): Training data.
            target_col (str): Name of the target column.
            group_col (str): Name of the grouping column.
        """
        groups = df[group_col].unique()
        for g in groups:
            subset = df[df[group_col] == g][target_col]
            mu = subset.mean()
            sigma = subset.std()

            # Handle constant values or single samples where std might be NaN or 0
            if pd.isna(sigma) or sigma == 0:
                sigma = 1.0

            self.means[str(g)] = mu
            self.stds[str(g)] = sigma

        self.fitted = True

    def save(self, path):
        """
        Saves the internal statistics to a .npy file.
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before saving.")

        # Save as a dictionary wrapped in a numpy object
        stats = {"means": self.means, "stds": self.stds}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, stats)

    def load(self, path):
        """
        Loads statistics from a .npy file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler stats file not found at {path}")

        stats = np.load(path, allow_pickle=True).item()
        self.means = stats["means"]
        self.stds = stats["stds"]
        self.fitted = True

    def fit_or_load(self, df, target_col, group_col, save_path, load_cached_data=True):
        """
        Tries to load stats from cache. If not found or forced, fits on data and saves.
        Implements the deterministic data processing caching logic.

        Args:
            df (pd.DataFrame): Data to fit on if cache is missing.
            target_col (str): Target column name.
            group_col (str): Group column name.
            save_path (str): Path to save/load stats.
            load_cached_data (bool): Whether to attempt loading from cache.
        """
        if load_cached_data and os.path.exists(save_path):
            try:
                self.load(save_path)
                # Simple validation: check if groups in df are covered
                # We convert groups to strings to match keys
                groups = df[group_col].unique().astype(str)
                if all(g in self.means for g in groups):
                    return
            except Exception:
                # If loading fails or validation fails, fall through to fit
                pass

        self.fit(df, target_col, group_col)
        self.save(save_path)

    def transform(self, values, groups):
        """
        Standardizes values based on their group: z = (x - mu) / sigma

        Args:
            values (np.ndarray or torch.Tensor): Values to transform.
            groups (list or np.ndarray): Group labels corresponding to values.

        Returns:
            Transformed values (same type as input).
        """
        is_torch = torch.is_tensor(values)
        if is_torch:
            device = values.device
            values_np = values.detach().cpu().numpy()
        else:
            values_np = np.array(values)

        # Ensure groups are strings for lookup
        groups = np.array(groups).astype(str)

        # Vectorized lookup
        # Using list comprehension is efficient enough for batch sizes (e.g. 128)
        mus = np.array([self.means.get(g, 0.0) for g in groups])
        sigmas = np.array([self.stds.get(g, 1.0) for g in groups])

        scaled = (values_np - mus) / sigmas

        if is_torch:
            return torch.tensor(scaled, dtype=values.dtype, device=device)
        return scaled

    def inverse_transform(self, values, groups):
        """
        Reverts standardization: x = z * sigma + mu
        """
        is_torch = torch.is_tensor(values)
        if is_torch:
            device = values.device
            values_np = values.detach().cpu().numpy()
        else:
            values_np = np.array(values)

        groups = np.array(groups).astype(str)

        mus = np.array([self.means.get(g, 0.0) for g in groups])
        sigmas = np.array([self.stds.get(g, 1.0) for g in groups])

        original = (values_np * sigmas) + mus

        if is_torch:
            return torch.tensor(original, dtype=values.dtype, device=device)
        return original


class SimpleScaler:
    """
    Basic StandardScaler (Global Mean/Std) for auxiliary targets like Shielding or Charges.
    """

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.fitted = False

    def fit(self, values):
        self.mean = np.mean(values)
        self.std = np.std(values)
        if self.std == 0:
            self.std = 1.0
        self.fitted = True

    def transform(self, values):
        if torch.is_tensor(values):
            return (values - self.mean) / self.std
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        if torch.is_tensor(values):
            return (values * self.std) + self.mean
        return (values * self.std) + self.mean


def mean_log_mae(y_true, y_pred, types):
    """
    Calculates the Log of the Mean Absolute Error, calculated for each scalar coupling type,
    and then averaged across types.

    Metric = 1/T * sum_{t} ( log( 1/n_t * sum |y_true - y_pred| ) )

    Args:
        y_true: Ground truth values (Tensor or numpy array)
        y_pred: Predicted values (Tensor or numpy array)
        types: 1D array/list of coupling types corresponding to inputs

    Returns:
        float: The LMAE score.
    """
    # Convert tensors to numpy
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()

    # Flatten arrays
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    types = np.array(types).flatten()

    unique_types = np.unique(types)
    log_maes = []

    for t in unique_types:
        mask = types == t
        if not np.any(mask):
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        # Calculate MAE for this type
        mae = np.mean(np.abs(y_t - y_p))

        # Log of MAE (add epsilon for numerical stability)
        # We use a small epsilon to avoid log(0) in case of perfect prediction
        log_mae = np.log(mae + 1e-9)
        log_maes.append(log_mae)

    if not log_maes:
        return 0.0

    return np.mean(log_maes)


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking loss during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
