import torch
import torch.nn as nn
import numpy as np
import random
import os
from library.config import Config


def set_seed(seed: int = Config.RANDOM_SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GaussianRBF(nn.Module):
    """
    Expands scalar distances into a vector of Gaussian Radial Basis Function features.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        n_centers: int = 60,
        width: float = None,
    ):
        super().__init__()
        self.start = start
        self.stop = stop
        self.n_centers = n_centers

        # Compute centers linearly spaced between start and stop
        centers = torch.linspace(start, stop, n_centers)
        self.register_buffer("centers", centers)

        # If width is not provided, estimate it based on the spacing between centers
        if width is None:
            width = (stop - start) / (n_centers - 1)
        self.width = width

        # Gamma parameter for the Gaussian function: exp(-gamma * (x - mu)^2)
        # We choose gamma such that the basis functions overlap reasonably.
        self.gamma = 1.0 / (self.width**2)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: Tensor of shape (..., ) containing interatomic distances.

        Returns:
            Tensor of shape (..., n_centers) containing RBF features.
        """
        # distances: [N] -> [N, 1]
        # centers: [n_centers] -> [1, n_centers]
        # result: [N, n_centers]
        diff = distances.unsqueeze(-1) - self.centers.unsqueeze(0)
        return torch.exp(-self.gamma * (diff**2))


class TargetScaler:
    """
    Standardizes target variables by removing the mean and scaling to unit variance.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, y: torch.Tensor):
        """
        Computes the mean and standard deviation to be used for later scaling.

        Args:
            y: Tensor of shape (N, D) or (N,) containing target values.
        """
        self.mean = torch.mean(y, dim=0)
        self.std = torch.std(y, dim=0)

        # Handle constant values to avoid division by zero
        if self.std.ndim == 0:
            if self.std == 0:
                self.std = torch.tensor(1.0)
        else:
            self.std[self.std == 0] = 1.0

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """
        Standardizes the data.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted yet.")
        return (y - self.mean) / self.std

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        """
        Scales the data back to the original representation.
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetScaler has not been fitted yet.")
        return y * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
