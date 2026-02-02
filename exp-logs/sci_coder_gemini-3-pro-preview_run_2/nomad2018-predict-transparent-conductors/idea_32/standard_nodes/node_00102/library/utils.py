import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = 42):
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


def rmsle(y_true, y_pred):
    """
    Calculates the Mean Column-wise Root Mean Squared Logarithmic Error.

    Args:
        y_true: Ground truth values (numpy array or torch tensor).
        y_pred: Predicted values (numpy array or torch tensor).

    Returns:
        float: The mean of the RMSLE calculated for each column.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Clip values to be non-negative to avoid errors in log
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    # Calculate log(x + 1)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    # Squared differences
    squared_error = (log_true - log_pred) ** 2

    # Calculate RMSLE for each column (target variable) separately
    if squared_error.ndim > 1:
        # Mean over samples (axis 0) -> (num_targets,)
        mean_squared_error_per_col = np.mean(squared_error, axis=0)
        rmsle_per_col = np.sqrt(mean_squared_error_per_col)
        # Return the average RMSLE across all targets
        return np.mean(rmsle_per_col)
    else:
        # Scalar case
        return np.sqrt(np.mean(squared_error))


class TargetScaler:
    """
    Standardizes regression targets by removing the mean and scaling to unit variance.
    Supports saving and loading state to/from .npz files for inference.
    """

    def __init__(self):
        self.mean = None
        self.std = None
        self.device = Config.DEVICE

    def fit(self, y):
        """
        Computes mean and standard deviation from the training targets.

        Args:
            y (torch.Tensor or np.ndarray): Training targets.
        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        # Move to device for calculation
        y = y.to(self.device)

        self.mean = torch.mean(y, dim=0)
        self.std = torch.std(y, dim=0)

        # Handle constant columns to avoid division by zero
        self.std[self.std == 0] = 1.0

    def transform(self, y):
        """
        Standardizes the input data using the fitted mean and std.

        Args:
            y (torch.Tensor or np.ndarray): Data to transform.

        Returns:
            torch.Tensor: Standardized data on the configured device.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler must be fitted before transform.")

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        y = y.to(self.device)
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        """
        Scales the data back to the original representation.

        Args:
            y (torch.Tensor or np.ndarray): Standardized data.

        Returns:
            torch.Tensor: Data in original scale on the configured device.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler must be fitted before inverse_transform.")

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        y = y.to(self.device)
        return y * self.std + self.mean

    def save(self, path):
        """
        Saves the scaler state (mean and std) to a .npz file.

        Args:
            path (str): File path to save the state.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler is not fitted, cannot save state.")

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            mean=self.mean.detach().cpu().numpy(),
            std=self.std.detach().cpu().numpy(),
        )
        print(f"TargetScaler state saved to {path}")

    def load(self, path):
        """
        Loads the scaler state (mean and std) from a .npz file.

        Args:
            path (str): File path to load the state from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Scaler state file not found at {path}")

        data = np.load(path)
        self.mean = torch.from_numpy(data["mean"]).to(self.device)
        self.std = torch.from_numpy(data["std"]).to(self.device)
        print(f"TargetScaler state loaded from {path}")
