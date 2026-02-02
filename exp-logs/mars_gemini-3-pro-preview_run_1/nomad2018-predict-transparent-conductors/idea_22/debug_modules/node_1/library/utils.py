import os
import random
import json
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
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


def get_device() -> torch.device:
    """
    Returns the appropriate torch device (CUDA if available, else CPU).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Root Mean Squared Logarithmic Error.

    Args:
        y_true: Ground truth values (original scale).
        y_pred: Predicted values (original scale).

    Returns:
        The RMSLE score.
    """
    # Ensure no negative values are passed to log
    y_true_clipped = np.maximum(y_true, 0)
    y_pred_clipped = np.maximum(y_pred, 0)

    log_true = np.log1p(y_true_clipped)
    log_pred = np.log1p(y_pred_clipped)

    squared_error = np.square(log_pred - log_true)
    mean_squared_error = np.mean(squared_error)

    return np.sqrt(mean_squared_error)


class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    Supports saving and loading state for consistent inference.
    """

    def __init__(self):
        self.mean = None
        self.scale = None
        self.epsilon = 1e-8

    def fit(self, X: np.ndarray):
        """
        Compute the mean and std to be used for later scaling.
        """
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        # Avoid division by zero
        self.scale = np.where(self.scale < self.epsilon, 1.0, self.scale)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.
        """
        if self.mean is None or self.scale is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")
        return (X - self.mean) / self.scale

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale back the data to the original representation.
        """
        if self.mean is None or self.scale is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")
        return (X * self.scale) + self.mean

    def save(self, path: str):
        """
        Save the mean and scale parameters to a JSON file.
        """
        if self.mean is None or self.scale is None:
            raise RuntimeError("StandardScaler has not been fitted yet.")

        data = {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        """
        Load the mean and scale parameters from a JSON file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        with open(path, "r") as f:
            data = json.load(f)

        self.mean = np.array(data["mean"])
        self.scale = np.array(data["scale"])
        return self
