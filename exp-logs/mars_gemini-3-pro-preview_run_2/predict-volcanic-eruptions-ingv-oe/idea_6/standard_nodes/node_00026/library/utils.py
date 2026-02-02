import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Wraps the implementation provided in Config.
    """
    Config.seed_everything(seed)


class TargetScaler:
    """
    Wraps sklearn's StandardScaler to handle the scaling of the 'time_to_eruption' target.
    Handles reshaping of 1D arrays and persistence of scaler statistics (mean, std)
    to ensuring consistent scaling between training and inference.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, y):
        """
        Compute the mean and std to be used for later scaling.

        Args:
            y (array-like): The target values (time_to_eruption).
        """
        y = np.array(y).reshape(-1, 1)
        self.scaler.fit(y)
        return self

    def transform(self, y):
        """
        Perform standardization by centering and scaling.

        Args:
            y (array-like): The data to scale.

        Returns:
            np.ndarray: The scaled data (1D array).
        """
        y = np.array(y).reshape(-1, 1)
        return self.scaler.transform(y).flatten()

    def fit_transform(self, y):
        """
        Fit to data, then transform it.

        Args:
            y (array-like): The target values.

        Returns:
            np.ndarray: The scaled data (1D array).
        """
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        """
        Scale back the data to the original representation.

        Args:
            y (array-like): The scaled data.

        Returns:
            np.ndarray: The data in original scale (1D array).
        """
        y = np.array(y).reshape(-1, 1)
        return self.scaler.inverse_transform(y).flatten()

    def save(self, mean_path: str, std_path: str):
        """
        Save the scaler statistics (mean and scale) to disk using numpy .npy format.

        Args:
            mean_path (str): Path to save the mean value.
            std_path (str): Path to save the scale (std) value.
        """
        if not hasattr(self.scaler, "mean_") or not hasattr(self.scaler, "scale_"):
            raise RuntimeError("Scaler has not been fitted. Call fit() before saving.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(mean_path), exist_ok=True)
        os.makedirs(os.path.dirname(std_path), exist_ok=True)

        np.save(mean_path, self.scaler.mean_)
        np.save(std_path, self.scaler.scale_)

    def load(self, mean_path: str, std_path: str):
        """
        Load scaler statistics from disk and initialize the internal StandardScaler.

        Args:
            mean_path (str): Path to the saved mean .npy file.
            std_path (str): Path to the saved scale .npy file.
        """
        if not os.path.exists(mean_path):
            raise FileNotFoundError(f"Mean file not found at {mean_path}")
        if not os.path.exists(std_path):
            raise FileNotFoundError(f"Std file not found at {std_path}")

        mean_val = np.load(mean_path)
        scale_val = np.load(std_path)

        self.scaler.mean_ = mean_val
        self.scaler.scale_ = scale_val
        self.scaler.var_ = scale_val**2

        return self
