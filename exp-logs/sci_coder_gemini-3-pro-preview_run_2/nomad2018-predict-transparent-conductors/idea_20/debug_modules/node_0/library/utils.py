import torch
import numpy as np
import random
import os


def set_seed(seed=42):
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
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_rmsle(y_pred, y_true):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): Ground truth values.

    Returns:
        float: The average RMSLE across columns.
    """
    # Detach and move to CPU for calculation to avoid graph retention
    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu()

    # Ensure non-negative predictions for log calculation
    y_pred = torch.clamp(y_pred, min=0.0)

    # Compute log(x + 1)
    log_pred = torch.log1p(y_pred)
    log_true = torch.log1p(y_true)

    # Compute squared error
    squared_error = (log_pred - log_true) ** 2

    # Mean squared error per column (dimension 0 is batch)
    mse_per_col = torch.mean(squared_error, dim=0)

    # Root mean squared error per column
    rmsle_per_col = torch.sqrt(mse_per_col)

    # Average across columns to get the final metric
    return torch.mean(rmsle_per_col).item()


class StandardScaler:
    """
    A utility class for standardizing data (zero mean, unit variance) using PyTorch tensors.
    """

    def __init__(self, device="cpu"):
        self.mean = None
        self.std = None
        self.device = device

    def fit(self, data):
        """
        Computes the mean and standard deviation of the data.

        Args:
            data (torch.Tensor): Input data of shape (N, features).
        """
        # Ensure data is on the correct device for calculation, though usually fits are done on CPU or GPU
        # depending on dataset size. We'll use the scaler's device.
        if data.device != self.device:
            data = data.to(self.device)

        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

        # Handle constant features (std=0) to avoid division by zero
        # Replace 0s with 1s so division doesn't change the value (0/1 = 0)
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        """
        Standardizes the data using the fitted mean and std.

        Args:
            data (torch.Tensor): Data to transform.

        Returns:
            torch.Tensor: Standardized data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "StandardScaler must be fitted before calling transform."
            )

        # Ensure data is on the correct device
        data = data.to(self.device)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Inverse transforms the standardized data back to the original scale.

        Args:
            data (torch.Tensor): Standardized data.

        Returns:
            torch.Tensor: Data in original scale.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError(
                "StandardScaler must be fitted before calling inverse_transform."
            )

        # Ensure data is on the correct device
        data = data.to(self.device)
        return (data * self.std) + self.mean

    def to(self, device):
        """
        Moves the scaler statistics to the specified device.

        Args:
            device (torch.device or str): The target device.

        Returns:
            StandardScaler: self
        """
        self.device = device
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self

    def state_dict(self):
        """Returns the state of the scaler as a dictionary."""
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        """Loads the state of the scaler from a dictionary."""
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]

        # Move loaded tensors to the current device
        if self.mean is not None:
            self.mean = self.mean.to(self.device)
        if self.std is not None:
            self.std = self.std.to(self.device)
