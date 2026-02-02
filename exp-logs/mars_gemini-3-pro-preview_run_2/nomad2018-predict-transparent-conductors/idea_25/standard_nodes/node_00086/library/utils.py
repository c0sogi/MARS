import os
import random
import numpy as np
import torch
import pandas as pd
from library.config import CACHE_DIR


def set_seed(seed):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_rmsle(y_pred, y_true):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_pred: Predicted values (Tensor or NumPy array).
        y_true: Ground truth values (Tensor or NumPy array).

    Returns:
        float: The mean RMSLE across all columns.
    """
    # Convert to numpy if tensors
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Ensure non-negative values for log
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # Compute RMSLE for each column
    log_pred = np.log1p(y_pred)
    log_true = np.log1p(y_true)
    squared_error = (log_pred - log_true) ** 2
    mean_squared_error = np.mean(squared_error, axis=0)
    rmsle_per_column = np.sqrt(mean_squared_error)

    # Return mean across columns
    return np.mean(rmsle_per_column)


class StandardScaler:
    """
    StandardScaler that normalizes data to zero mean and unit variance.
    Supports PyTorch Tensors and NumPy arrays.
    """

    def __init__(self, mean=None, std=None, device=None):
        self.mean = mean
        self.std = std
        self.device = device

    def fit(self, data):
        """
        Computes mean and std from the data.
        """
        if isinstance(data, torch.Tensor):
            self.mean = torch.mean(data, dim=0)
            self.std = torch.std(data, dim=0)
            self.device = data.device
        else:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)

    def transform(self, data):
        """
        Standardizes the data using computed mean and std.
        """
        # Handle epsilon for stability if std is 0 (unlikely for regression targets but good practice)
        std_safe = self.std + 1e-8

        if isinstance(data, torch.Tensor):
            # Ensure mean/std are on the same device as data
            if self.mean.device != data.device:
                self.mean = self.mean.to(data.device)
                self.std = self.std.to(data.device)
            return (data - self.mean) / std_safe
        else:
            return (data - self.mean) / std_safe

    def inverse_transform(self, data):
        """
        Scales the data back to original range.
        """
        if isinstance(data, torch.Tensor):
            if self.mean.device != data.device:
                self.mean = self.mean.to(data.device)
                self.std = self.std.to(data.device)
            return (data * self.std) + self.mean
        else:
            return (data * self.std) + self.mean

    def save(self, path):
        """
        Saves the mean and std to a .npz file (no pickle).
        """
        # Convert to numpy for saving
        mean_np = (
            self.mean.cpu().numpy()
            if isinstance(self.mean, torch.Tensor)
            else self.mean
        )
        std_np = (
            self.std.cpu().numpy() if isinstance(self.std, torch.Tensor) else self.std
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=mean_np, std=std_np)

    def load(self, path):
        """
        Loads mean and std from a .npz file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = torch.from_numpy(data["mean"])
        self.std = torch.from_numpy(data["std"])
        # By default load to CPU, move to device during transform if needed


def get_scaler(data, cache_path, load_cached_data=True):
    """
    Retrieves a fitted StandardScaler.
    Follows the deterministic caching logic:
    1. IF load_cached_data is True: Try to load from cache_path.
    2. IF loading fails OR load_cached_data is False: Fit on data and save.

    Args:
        data: Data to fit on (if cache miss).
        cache_path: Path to save/load the scaler.
        load_cached_data: Boolean flag to enable/disable loading.

    Returns:
        Fitted StandardScaler object.
    """
    scaler = StandardScaler()

    # Logic Flow
    loaded = False
    if load_cached_data:
        if os.path.exists(cache_path):
            try:
                scaler.load(cache_path)
                loaded = True
                # print(f"Loaded scaler from {cache_path}")
            except Exception as e:
                print(f"Failed to load scaler: {e}. Recomputing.")
                loaded = False
        else:
            # print(f"Scaler cache not found at {cache_path}. Recomputing.")
            loaded = False

    if not loaded:
        scaler.fit(data)
        scaler.save(cache_path)
        # print(f"Fitted scaler and saved to {cache_path}")

    return scaler
