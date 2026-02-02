import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = Config.SEED) -> None:
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
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


def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the mean column-wise Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_true (np.ndarray): Ground truth values of shape (N, D).
        y_pred (np.ndarray): Predicted values of shape (N, D).

    Returns:
        float: The mean of the RMSLE calculated for each column.
    """
    # Clip predictions to be non-negative as log is undefined for negative numbers
    # Formation energy can theoretically be 0, bandgap > 0.
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # Calculate log(1 + x)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    # Calculate squared differences
    squared_error = (log_pred - log_true) ** 2

    # Calculate Mean Squared Error for each column
    mse_per_column = np.mean(squared_error, axis=0)

    # Calculate Root Mean Squared Error for each column
    rmsle_per_column = np.sqrt(mse_per_column)

    # Return the average of the column-wise RMSLEs
    return np.mean(rmsle_per_column)


class TargetScaler:
    """
    A class to standardize target variables (zero mean, unit variance) and
    inverse-transform them back to the original scale.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y: np.ndarray) -> None:
        """
        Computes mean and standard deviation of the target variables.

        Args:
            y (np.ndarray): The target data to fit, shape (N, D).
        """
        self.mean = np.mean(y, axis=0)
        self.std = np.std(y, axis=0)

        # Avoid division by zero if std is 0 (constant feature)
        # Replace 0 with 1 to leave those features unchanged during division
        self.std[self.std < 1e-8] = 1.0

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Standardizes the data using the fitted mean and standard deviation.

        Args:
            y (np.ndarray): The data to transform.

        Returns:
            np.ndarray: The standardized data.
        """
        if self.mean is None or self.std is None:
            raise ValueError(
                "Scaler has not been fitted yet. Call fit() or load() first."
            )
        return (y - self.mean) / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        """
        Inverse standardizes the data to the original scale.

        Args:
            y (np.ndarray): The standardized data.

        Returns:
            np.ndarray: The data in original scale.
        """
        if self.mean is None or self.std is None:
            raise ValueError(
                "Scaler has not been fitted yet. Call fit() or load() first."
            )
        return (y * self.std) + self.mean

    def save(self, path: str) -> None:
        """
        Saves the scaler parameters (mean and std) to a .npz file.

        Args:
            path (str): The file path to save the scaler parameters.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path: str) -> None:
        """
        Loads the scaler parameters from a .npz file.

        Args:
            path (str): The file path to load the scaler parameters from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]
