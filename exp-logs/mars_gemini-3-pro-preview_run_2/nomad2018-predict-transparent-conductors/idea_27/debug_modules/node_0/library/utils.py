import random
import os
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class StandardScaler:
    """
    Utility class for standardizing target variables (zero mean, unit variance).
    Supports saving and loading state to/from .npz files.
    """

    def __init__(self, device=Config.DEVICE):
        self.mean = None
        self.std = None
        self.device = device

    def fit(self, y):
        """
        Computes mean and std from the training targets.

        Args:
            y (torch.Tensor or np.ndarray): Target values of shape (N, num_targets).
        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        y = y.to(self.device)
        self.mean = torch.mean(y, dim=0)
        self.std = torch.std(y, dim=0)

        # Handle zero standard deviation to avoid division by zero
        self.std = torch.where(
            self.std == 0, torch.tensor(1.0, device=self.device), self.std
        )

    def transform(self, y):
        """
        Standardizes the input tensor using fitted mean and std.

        Args:
            y (torch.Tensor or np.ndarray): Target values to standardize.

        Returns:
            torch.Tensor: Standardized values.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        y = y.to(self.device)
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        """
        Reverses the standardization to get original scale values.

        Args:
            y (torch.Tensor or np.ndarray): Standardized values.

        Returns:
            torch.Tensor: Values in original scale.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        y = y.to(self.device)
        return y * self.std + self.mean

    def save(self, path):
        """
        Saves the scaler state (mean and std) to a .npz file.

        Args:
            path (str): Path to save the .npz file.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        np.savez(path, mean=self.mean.cpu().numpy(), std=self.std.cpu().numpy())

    def load(self, path):
        """
        Loads the scaler state from a .npz file.

        Args:
            path (str): Path to the .npz file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = torch.from_numpy(data["mean"]).to(self.device)
        self.std = torch.from_numpy(data["std"]).to(self.device)


def compute_metric(y_true, y_pred):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values.
        y_pred (torch.Tensor or np.ndarray): Predicted values.

    Returns:
        float: The mean of the column-wise RMSLE.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure non-negative values for log calculation
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    squared_error = (log_true - log_pred) ** 2
    mse_per_col = np.mean(squared_error, axis=0)
    rmsle_per_col = np.sqrt(mse_per_col)

    return np.mean(rmsle_per_col)
