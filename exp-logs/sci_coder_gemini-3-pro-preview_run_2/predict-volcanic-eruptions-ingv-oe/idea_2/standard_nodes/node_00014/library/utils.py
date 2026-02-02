import os
import numpy as np
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility by delegating to Config.set_seed.

    Args:
        seed (int): The seed value to use.
    """
    Config.set_seed(seed)


class TargetScaler:
    """
    Handles Standard Scaling (Z-score normalization) for the target variable.
    Crucial for stabilizing regression loss when targets have large magnitudes.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y):
        """
        Computes mean and std from the target array y.

        Args:
            y (np.array or pd.Series): Target values.
        """
        self.mean = np.mean(y)
        self.std = np.std(y)
        # Avoid division by zero
        if self.std == 0:
            self.std = 1.0

    def transform(self, y):
        """
        Scales the input y using the computed mean and std.

        Args:
            y (np.array or pd.Series): Target values to scale.

        Returns:
            np.array: Scaled values.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")
        return (y - self.mean) / self.std

    def inverse_transform(self, y_scaled):
        """
        Converts scaled values back to the original scale.

        Args:
            y_scaled (np.array, float, or torch.Tensor): Scaled values.

        Returns:
            np.array or float: Values in original scale.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        # Handle torch tensors seamlessly
        if hasattr(y_scaled, "detach"):
            y_scaled = y_scaled.detach().cpu().numpy()

        return (y_scaled * self.std) + self.mean

    def save(self, mean_path, std_path):
        """
        Saves the mean and std to .npy files.

        Args:
            mean_path (str): File path to save the mean.
            std_path (str): File path to save the std.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        # Ensure parent directories exist
        os.makedirs(os.path.dirname(mean_path), exist_ok=True)
        os.makedirs(os.path.dirname(std_path), exist_ok=True)

        np.save(mean_path, self.mean)
        np.save(std_path, self.std)

    def load(self, mean_path, std_path):
        """
        Loads the mean and std from .npy files.

        Args:
            mean_path (str): File path to load the mean from.
            std_path (str): File path to load the std from.
        """
        if not os.path.exists(mean_path) or not os.path.exists(std_path):
            raise FileNotFoundError(f"Scaler files not found: {mean_path}, {std_path}")

        self.mean = np.load(mean_path)
        self.std = np.load(std_path)
