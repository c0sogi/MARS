import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    Sets the random seed for python, numpy, and torch to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TargetScaler:
    """
    A utility class to standardize target variables (zero mean, unit variance).
    It supports saving and loading the scaling parameters (mean, std) using numpy formats
    to avoid pickle, ensuring compatibility and safety.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y):
        """
        Compute the mean and standard deviation of the targets.

        Args:
            y (torch.Tensor): A tensor of shape (N, num_targets) containing the target values.
        """
        # Calculate mean and std along the batch dimension (dim 0)
        self.mean = torch.mean(y, dim=0)
        self.std = torch.std(y, dim=0)

        # Handle cases where std is 0 (constant target) to avoid division by zero
        # Replace 0 with 1 to leave those values unchanged during division
        self.std = torch.where(
            self.std == 0, torch.tensor(1.0, device=y.device), self.std
        )

    def transform(self, y):
        """
        Standardize the targets using the fitted mean and std.
        z = (y - mean) / std

        Args:
            y (torch.Tensor): Target values to standardize.

        Returns:
            torch.Tensor: Standardized targets.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted yet.")

        # Ensure scaler parameters are on the same device as the input
        if self.mean.device != y.device:
            self.mean = self.mean.to(y.device)
            self.std = self.std.to(y.device)

        return (y - self.mean) / self.std

    def inverse_transform(self, y_scaled):
        """
        Convert standardized targets back to their original scale.
        y = z * std + mean

        Args:
            y_scaled (torch.Tensor): Standardized target values.

        Returns:
            torch.Tensor: Targets in original scale.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted yet.")

        # Ensure scaler parameters are on the same device as the input
        if self.mean.device != y_scaled.device:
            self.mean = self.mean.to(y_scaled.device)
            self.std = self.std.to(y_scaled.device)

        return (y_scaled * self.std) + self.mean

    def save(self, path):
        """
        Save the mean and std to a .npz file.

        Args:
            path (str): The file path to save the parameters.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted yet.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Convert to numpy for saving
        mean_np = self.mean.detach().cpu().numpy()
        std_np = self.std.detach().cpu().numpy()

        np.savez(path, mean=mean_np, std=std_np)

    def load(self, path):
        """
        Load the mean and std from a .npz file.

        Args:
            path (str): The file path to load the parameters from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = torch.from_numpy(data["mean"])
        self.std = torch.from_numpy(data["std"])
