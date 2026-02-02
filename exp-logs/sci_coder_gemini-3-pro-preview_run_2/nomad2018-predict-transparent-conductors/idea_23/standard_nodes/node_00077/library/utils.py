import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
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


class TargetScaler:
    """
    A utility class to standardize target variables (zero mean, unit variance)
    and inverse transform predictions. Supports saving and loading state
    using numpy files.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y):
        """
        Computes the mean and standard deviation of the targets.

        Args:
            y (np.ndarray): Target values of shape (N, num_targets).
        """
        self.mean = np.mean(y, axis=0)
        self.std = np.std(y, axis=0)
        # Prevent division by zero by replacing 0 std with 1.0
        # This handles constant target columns if any exist
        self.std[self.std == 0] = 1.0

    def transform(self, y):
        """
        Standardizes the targets using the fitted mean and std.

        Args:
            y (np.ndarray): Target values.

        Returns:
            np.ndarray: Standardized targets.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted yet.")
        return (y - self.mean) / self.std

    def inverse_transform(self, y_scaled):
        """
        Reverts the standardization to original scale.

        Args:
            y_scaled (np.ndarray): Standardized predictions.

        Returns:
            np.ndarray: Predictions in original scale.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted yet.")
        return y_scaled * self.std + self.mean

    def save(self, path):
        """
        Saves the scaler parameters to a .npz file.

        Args:
            path (str): File path to save the scaler state.
        """
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        """
        Loads the scaler parameters from a .npz file.

        Args:
            path (str): File path to load the scaler state from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler cache file not found at {path}")

        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
