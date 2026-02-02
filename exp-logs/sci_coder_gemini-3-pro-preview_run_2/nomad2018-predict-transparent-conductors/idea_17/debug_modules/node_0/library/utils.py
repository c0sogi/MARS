import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python's random, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class StandardScaler:
    """
    A utility class for standardizing data (zero mean, unit variance).
    Supports PyTorch tensors and saving/loading state to/from disk.
    """

    def __init__(self, mean=None, std=None, device=None):
        self.mean = mean
        self.std = std
        self.device = device

        if self.mean is not None and self.device is not None:
            self.mean = self.mean.to(self.device)
        if self.std is not None and self.device is not None:
            self.std = self.std.to(self.device)

    def fit(self, data):
        """
        Computes the mean and standard deviation of the provided data.

        Args:
            data (torch.Tensor or np.ndarray): The data to fit, shape (N, features).
        """
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)

        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

        # Handle constant features by replacing 0 std with 1.0 to avoid division by zero
        self.std = torch.where(
            self.std == 0, torch.tensor(1.0, device=data.device), self.std
        )

        if self.device:
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)

    def transform(self, data):
        """
        Standardizes the data using the fitted mean and std.

        Args:
            data (torch.Tensor or np.ndarray): Data to transform.

        Returns:
            torch.Tensor: Standardized data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")

        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)

        # Ensure data and stats are on the same device
        target_device = data.device
        if self.mean.device != target_device:
            self.mean = self.mean.to(target_device)
            self.std = self.std.to(target_device)

        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Reverts the standardization (scales back to original range).

        Args:
            data (torch.Tensor or np.ndarray): Standardized data.

        Returns:
            torch.Tensor: Data in original scale.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")

        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)

        target_device = data.device
        if self.mean.device != target_device:
            self.mean = self.mean.to(target_device)
            self.std = self.std.to(target_device)

        return (data * self.std) + self.mean

    def save(self, path):
        """
        Saves the mean and std to a .npz file.

        Args:
            path (str): File path to save the scaler state.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot save unfitted StandardScaler.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez(
            path,
            mean=self.mean.detach().cpu().numpy(),
            std=self.std.detach().cpu().numpy(),
        )

    def load(self, path):
        """
        Loads mean and std from a .npz file.

        Args:
            path (str): File path to load the scaler state from.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not os.path.exists(path):
            return False

        try:
            data = np.load(path)
            self.mean = torch.from_numpy(data["mean"])
            self.std = torch.from_numpy(data["std"])

            if self.device:
                self.mean = self.mean.to(self.device)
                self.std = self.std.to(self.device)
            return True
        except Exception as e:
            # In a real scenario, logging this error would be good practice
            return False


def get_scaler(data, cache_path, load_cached_data=True, device=None):
    """
    Retrieves a StandardScaler, handling caching logic.

    Logic Flow:
    1. IF load_cached_data is True: Try to load from cache_path.
    2. IF loading fails OR load_cached_data is False:
       - Fit on the provided data.
       - Save the fitted state to cache_path.

    Args:
        data (torch.Tensor or np.ndarray): Data to fit the scaler on (if cache miss).
        cache_path (str): Path to the cached scaler file.
        load_cached_data (bool): Whether to attempt loading from cache.
        device (torch.device): Device to place the scaler stats on.

    Returns:
        StandardScaler: A fitted scaler instance.
    """
    scaler = StandardScaler(device=device)

    loaded = False
    if load_cached_data:
        loaded = scaler.load(cache_path)

    if not loaded:
        if data is None:
            raise ValueError(
                "Cache loading failed or disabled, but no data provided to fit scaler."
            )

        # Ensure directory exists before fitting/saving
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        scaler.fit(data)
        scaler.save(cache_path)

    return scaler
