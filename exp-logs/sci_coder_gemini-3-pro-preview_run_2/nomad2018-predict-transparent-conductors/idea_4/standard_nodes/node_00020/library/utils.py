import torch
import torch.nn as nn
import numpy as np


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Function expansion for scalar features.
    Expands a scalar value (e.g., distance, angle) into a vector of RBF activations.
    This enforces a strong geometric prior by encoding values based on their proximity
    to fixed centers.
    """

    def __init__(self, start, stop, n_centers, width=None):
        super().__init__()
        # Create centers linearly spaced between start and stop
        self.centers = torch.linspace(start, stop, n_centers)

        # Determine width of the RBFs
        if width is None:
            # Default width is the distance between adjacent centers
            if n_centers > 1:
                width = (stop - start) / (n_centers - 1)
            else:
                width = 1.0

        # Gamma parameter for the Gaussian function: exp(-gamma * x^2)
        # We use 1.0 / width^2 to control the spread
        self.gamma = 1.0 / (width**2)

        # Register buffers so they are part of the model state but not learnable parameters
        self.register_buffer("centers_buffer", self.centers)
        self.register_buffer("gamma_buffer", torch.tensor(self.gamma))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,) or (N, 1) containing scalar values.
        Returns:
            Tensor of shape (N, n_centers) containing RBF expansions.
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)  # (N, 1)

        # Calculate squared distance between inputs and centers
        # x: (N, 1), centers_buffer: (C,) -> Broadcasting results in (N, C)
        diff = x - self.centers_buffer

        # Compute Gaussian RBF
        rbf = torch.exp(-self.gamma_buffer * (diff**2))
        return rbf


class Standardizer:
    """
    Utility for standardizing targets (zero mean, unit variance).
    Helps in training stability by normalizing the output space.
    """

    def __init__(self, mean=None, std=None, device="cpu"):
        self.mean = mean
        self.std = std
        self.device = device

        # Move stats to device if provided during init
        if self.mean is not None:
            self.mean = self.mean.to(self.device)
        if self.std is not None:
            self.std = self.std.to(self.device)

    def fit(self, data):
        """
        Compute mean and std from training data.
        Args:
            data: Torch tensor of shape (N, D)
        """
        self.mean = torch.mean(data, dim=0).to(self.device)
        self.std = torch.std(data, dim=0).to(self.device)

        # Prevent division by zero
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        """
        Standardize data: (x - mean) / std
        """
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer must be fitted before transform.")
        data = data.to(self.device)
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        Un-standardize data: x * std + mean
        Used to convert model predictions back to original scale.
        """
        if self.mean is None or self.std is None:
            raise ValueError("Standardizer must be fitted before inverse_transform.")
        data = data.to(self.device)
        return (data * self.std) + self.mean

    def to(self, device):
        """
        Moves the internal statistics to the specified device.
        """
        self.device = device
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self


def compute_rmsle(y_true, y_pred):
    """
    Computes the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_true: Torch tensor or numpy array of ground truth values.
        y_pred: Torch tensor or numpy array of predicted values.

    Returns:
        float: The mean RMSLE averaged over all target columns.
    """
    # Convert numpy arrays to torch tensors if necessary
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Ensure predictions and targets are non-negative for log calculation
    # Clipping at 0.0 handles potential negative predictions from the model
    y_pred = torch.clamp(y_pred, min=0.0)
    y_true = torch.clamp(y_true, min=0.0)

    # Calculate squared logarithmic error element-wise
    # log1p(x) computes log(x + 1)
    squared_log_error = (torch.log1p(y_true) - torch.log1p(y_pred)) ** 2

    # Calculate Mean Squared Logarithmic Error for each column (dimension 0 is batch)
    mean_squared_log_error = torch.mean(squared_log_error, dim=0)

    # Calculate Root Mean Squared Logarithmic Error for each column
    rmsle_per_column = torch.sqrt(mean_squared_log_error)

    # Average the RMSLE across all columns (formation energy and bandgap)
    mean_rmsle = torch.mean(rmsle_per_column)

    return mean_rmsle.item()
