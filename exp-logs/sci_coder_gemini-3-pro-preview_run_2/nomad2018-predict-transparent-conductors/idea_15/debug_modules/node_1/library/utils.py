import os
import random
import numpy as np
import torch


def set_seed(seed):
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
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class Standardizer:
    """
    A utility class for Z-score normalization (standardization) of tensors.
    Can be used for both input features and target variables.
    Supports saving and loading the computed mean and standard deviation.
    """

    def __init__(self, device="cpu"):
        """
        Initialize the Standardizer.

        Args:
            device (str or torch.device): The device to store mean and std on.
        """
        self.mean = None
        self.std = None
        self.device = device
        self.epsilon = 1e-8

    def fit(self, data):
        """
        Computes the mean and standard deviation from the data.

        Args:
            data (torch.Tensor or np.ndarray): The data to fit on. Shape (N, D) or (N,).
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        # Ensure float
        data = data.float()

        self.mean = torch.mean(data, dim=0).to(self.device)
        self.std = torch.std(data, dim=0).to(self.device)

        # Handle cases where std is 0 (constant feature) to avoid division by zero
        self.std = torch.where(
            self.std < self.epsilon, torch.tensor(1.0, device=self.device), self.std
        )

        return self

    def transform(self, data):
        """
        Standardizes the data using the fitted mean and std.
        z = (x - mean) / std

        Args:
            data (torch.Tensor or np.ndarray): The data to transform.

        Returns:
            torch.Tensor: The standardized data on the specified device.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer has not been fitted yet.")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        data = data.float().to(self.device)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Reverses the standardization.
        x = z * std + mean

        Args:
            data (torch.Tensor or np.ndarray): The standardized data.

        Returns:
            torch.Tensor: The data in original scale on the specified device.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Standardizer has not been fitted yet.")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        data = data.float().to(self.device)
        return (data * self.std) + self.mean

    def save(self, path):
        """
        Saves the mean and standard deviation to a file using numpy format.

        Args:
            path (str): The file path to save to (e.g., 'scaler.npz').
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Cannot save unfitted Standardizer.")

        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        np.savez(
            path,
            mean=self.mean.detach().cpu().numpy(),
            std=self.std.detach().cpu().numpy(),
        )
        print(f"Standardizer state saved to {path}")

    def load(self, path):
        """
        Loads the mean and standard deviation from a file.

        Args:
            path (str): The file path to load from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Standardizer file not found at {path}")

        data = np.load(path)
        self.mean = torch.from_numpy(data["mean"]).to(self.device)
        self.std = torch.from_numpy(data["std"]).to(self.device)
        print(f"Standardizer state loaded from {path}")
        return self
