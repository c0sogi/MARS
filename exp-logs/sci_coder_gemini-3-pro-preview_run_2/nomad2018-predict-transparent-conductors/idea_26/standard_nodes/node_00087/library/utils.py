import random
import os
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    """
    Sets the seed for reproducibility across random, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RBFExpansion(nn.Module):
    """
    Expands distances into a Gaussian Radial Basis Function (RBF) set.
    """

    def __init__(self, vmin=0.0, vmax=8.0, bins=40, lengthscale=None):
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins

        # Centers of the Gaussian functions
        self.centers = torch.linspace(vmin, vmax, bins)

        # Determine gamma (1/sigma^2)
        if lengthscale is None:
            # If not provided, set gamma such that the width matches the bin spacing
            step = (vmax - vmin) / bins
            self.gamma = 1.0 / (step**2)
        else:
            self.gamma = lengthscale

        # Register centers as a buffer so it's part of the state_dict but not a parameter
        self.register_buffer("centers_buffer", self.centers)

    def forward(self, distance):
        """
        Args:
            distance: Tensor of shape (...) representing distances.
        Returns:
            Tensor of shape (..., bins) containing RBF expansion.
        """
        # distance: [N] -> [N, 1]
        # centers: [bins]
        # result: [N, bins]
        return torch.exp(
            -self.gamma * (distance.unsqueeze(-1) - self.centers_buffer) ** 2
        )


class StandardScaler:
    """
    Standardizes data by removing the mean and scaling to unit variance.
    Supports both PyTorch Tensors and NumPy arrays.
    """

    def __init__(self, mean=None, std=None, device=None):
        self.mean = mean
        self.std = std
        self.device = device

    def fit(self, data):
        """
        Computes the mean and std to be used for later scaling.
        Args:
            data: Tensor or numpy array of shape (N, features).
        """
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)

        # Compute stats along the batch dimension (dim 0)
        self.mean = torch.mean(data, dim=0)
        self.std = torch.std(data, dim=0)

        # Handle constant features (std=0) to avoid division by zero
        self.std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)

        if self.device:
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)

    def transform(self, data):
        """
        Perform standardization by centering and scaling.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        input_type_is_numpy = isinstance(data, np.ndarray)
        if input_type_is_numpy:
            data = torch.from_numpy(data)

        # Move stats to data device if needed
        if self.mean.device != data.device:
            self.mean = self.mean.to(data.device)
            self.std = self.std.to(data.device)

        scaled = (data - self.mean) / self.std

        if input_type_is_numpy:
            return scaled.cpu().numpy()
        return scaled

    def inverse_transform(self, data):
        """
        Scale back the data to the original representation.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        input_type_is_numpy = isinstance(data, np.ndarray)
        if input_type_is_numpy:
            data = torch.from_numpy(data)

        if self.mean.device != data.device:
            self.mean = self.mean.to(data.device)
            self.std = self.std.to(data.device)

        unscaled = data * self.std + self.mean

        if input_type_is_numpy:
            return unscaled.cpu().numpy()
        return unscaled

    def state_dict(self):
        return {
            "mean": self.mean.cpu() if self.mean is not None else None,
            "std": self.std.cpu() if self.std is not None else None,
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        if self.device and self.mean is not None:
            self.mean = self.mean.to(self.device)
            self.std = self.std.to(self.device)


def compute_metrics(preds, targets):
    """
    Computes Column-wise Root Mean Squared Logarithmic Error (RMSLE).
    Args:
        preds: Predictions (N, D)
        targets: Ground truth (N, D)
    Returns:
        float: Mean RMSLE across columns.
    """
    # Convert to numpy if tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure non-negative for log (physical quantities like energy/bandgap >= 0)
    # Clipping at 0 is a safe fallback for numerical stability
    preds = np.maximum(preds, 0)
    targets = np.maximum(targets, 0)

    # Compute Log(x + 1)
    log_preds = np.log1p(preds)
    log_targets = np.log1p(targets)

    # Squared Error in Log space
    squared_log_error = (log_preds - log_targets) ** 2

    # Mean Squared Log Error per column
    msle_per_col = np.mean(squared_log_error, axis=0)

    # Root Mean Squared Log Error per column
    rmsle_per_col = np.sqrt(msle_per_col)

    # Average across columns (metric is column-wise RMSLE)
    return np.mean(rmsle_per_col)
