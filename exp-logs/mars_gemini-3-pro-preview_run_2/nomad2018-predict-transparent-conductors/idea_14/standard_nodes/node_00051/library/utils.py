import torch
import numpy as np
import random
import os
from library.config import Config


def set_seed(seed=Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class StandardScaler:
    """
    Standardizes features by removing the mean and scaling to unit variance.
    Supports saving and loading state to/from disk using numpy.
    """

    def __init__(self, device=Config.DEVICE):
        self.mean = None
        self.std = None
        self.device = device

    def fit(self, data):
        """
        Compute the mean and std to be used for later scaling.
        Args:
            data: torch.Tensor or numpy.ndarray of shape (N, features)
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        # Move to device for calculation if needed, though usually fit is fast enough on CPU
        # Here we compute on the specified device to match stored state
        data = data.to(self.device)

        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

        # Handle constant features to avoid division by zero
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        """
        Perform standardization by centering and scaling.
        Args:
            data: torch.Tensor or numpy.ndarray
        Returns:
            torch.Tensor: Standardized data
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data = data.to(self.device)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Scale back the data to the original representation.
        Args:
            data: torch.Tensor or numpy.ndarray
        Returns:
            torch.Tensor: Original scale data
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        data = data.to(self.device)
        return (data * self.std) + self.mean

    def save(self, path):
        """
        Save the mean and std to a file using numpy.
        Args:
            path: File path to save the scaler state (.npz)
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            mean=self.mean.detach().cpu().numpy(),
            std=self.std.detach().cpu().numpy(),
        )

    def load(self, path):
        """
        Load the mean and std from a file.
        Args:
            path: File path to load the scaler state from (.npz)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler file not found at {path}")

        data = np.load(path)
        self.mean = torch.from_numpy(data["mean"]).to(self.device)
        self.std = torch.from_numpy(data["std"]).to(self.device)


def rmsle(y_true, y_pred):
    """
    Calculate the Column-wise Root Mean Squared Logarithmic Error (MCRMSE).

    Args:
        y_true: Ground truth values (torch.Tensor or numpy.ndarray)
        y_pred: Predicted values (torch.Tensor or numpy.ndarray)

    Returns:
        float: The mean of the RMSLE for each column.
    """
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    # Ensure data is on CPU for metric calculation to avoid sync overhead if not needed
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    # Clamp predictions to be non-negative as log is undefined for negative numbers
    y_pred = torch.clamp(y_pred, min=0.0)
    y_true = torch.clamp(y_true, min=0.0)

    log_true = torch.log1p(y_true)
    log_pred = torch.log1p(y_pred)

    # Calculate MSE per column (dim=0 is the batch dimension)
    mse_per_column = torch.mean((log_pred - log_true) ** 2, dim=0)

    # Calculate RMSE per column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Return the mean of column-wise RMSEs
    return torch.mean(rmse_per_column).item()
