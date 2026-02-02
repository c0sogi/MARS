import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Sets the seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TargetScaler:
    """
    Standardizes targets by removing the mean and scaling to unit variance (Z-score normalization).
    Handles saving and loading of scaler statistics using .npz format.
    """

    def __init__(self, device="cpu"):
        """
        Initialize the scaler.

        Args:
            device (str): The device ('cpu' or 'cuda') where tensors should be stored.
        """
        self.mean = None
        self.std = None
        self.device = device

    def fit(self, y):
        """
        Computes the mean and std to be used for later scaling.

        Args:
            y (torch.Tensor or np.ndarray): The data to fit, shape (N, num_targets).
        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        # Ensure y is on the correct device for calculation if needed,
        # though usually fit is done on CPU for stability before training.
        # We store parameters on the specified device.
        y_device = y.to(self.device)

        self.mean = torch.mean(y_device, dim=0)
        self.std = torch.std(y_device, dim=0)

        # Handle constant values to avoid division by zero
        # If std is 0, replace with 1 to leave values unchanged (centered but not scaled)
        self.std[self.std == 0] = 1.0

    def transform(self, y):
        """
        Standardizes the data.

        Args:
            y (torch.Tensor or np.ndarray): The data to transform.

        Returns:
            torch.Tensor: The scaled data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted yet.")

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        y = y.to(self.device)
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        """
        Scales the data back to the original representation.

        Args:
            y (torch.Tensor or np.ndarray): The scaled data.

        Returns:
            torch.Tensor: The data in original scale.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted yet.")

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        y = y.to(self.device)
        return y * self.std + self.mean

    def save(self, path):
        """
        Saves the scaler parameters (mean and std) to a .npz file.

        Args:
            path (str): The file path to save the scaler state.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted yet.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez(path, mean=self.mean.cpu().numpy(), std=self.std.cpu().numpy())

    def load(self, path):
        """
        Loads the scaler parameters from a .npz file.

        Args:
            path (str): The file path to load the scaler state from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = torch.from_numpy(data["mean"]).to(self.device)
        self.std = torch.from_numpy(data["std"]).to(self.device)
