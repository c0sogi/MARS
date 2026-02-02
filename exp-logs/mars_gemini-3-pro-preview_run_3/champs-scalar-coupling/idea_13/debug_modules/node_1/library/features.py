import torch
import torch.nn as nn
from library.config import Config


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Function (RBF) expansion for scalar distances.

    This layer expands a continuous distance value 'x' into a vector of 'num_rbf'
    Gaussian activations. This allows the neural network to learn non-linear
    relationships based on atomic distances (e.g., bond lengths).

    Formula: phi_k(x) = exp(-gamma * (x - mu_k)^2)
    where mu_k are centers uniformly spaced between 'start' and 'stop'.
    """

    def __init__(
        self,
        start=0.0,
        stop=Config.MAX_RADIUS,
        num_rbf=Config.NUM_RBF,
        gamma=Config.RBF_GAMMA,
    ):
        """
        Args:
            start (float): Minimum distance for the RBF centers.
            stop (float): Maximum distance for the RBF centers.
            num_rbf (int): Number of RBF centers (output dimension).
            gamma (float): The width/spread parameter of the Gaussian functions.
        """
        super().__init__()
        self.start = start
        self.stop = stop
        self.num_rbf = num_rbf
        self.gamma = gamma

        # Register centers as a buffer so they are part of the state_dict but not trainable parameters
        # Centers are uniformly spaced over the range [start, stop]
        centers = torch.linspace(start, stop, num_rbf)
        self.register_buffer("centers", centers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of distances. Shape (..., 1) or (...).

        Returns:
            torch.Tensor: Expanded RBF features. Shape (..., num_rbf).
        """
        # Ensure input has a last dimension for broadcasting
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Compute squared difference: (..., 1) - (num_rbf,) -> (..., num_rbf)
        diff = x - self.centers

        # Apply Gaussian function
        return torch.exp(-self.gamma * (diff**2))


class AngleRBF(nn.Module):
    """
    Gaussian Radial Basis Function (RBF) expansion for bond angles.

    This layer expands the cosine of the bond angle (cos theta) into a vector
    of Gaussian activations. This avoids the numerical instability of recursive
    basis functions while retaining expressivity for angular information.

    The input is expected to be in the range [-1, 1] (cosine values).
    """

    def __init__(
        self, start=-1.0, stop=1.0, num_rbf=Config.NUM_ANGLE_RBF, gamma=Config.RBF_GAMMA
    ):
        """
        Args:
            start (float): Minimum cosine value (usually -1.0).
            stop (float): Maximum cosine value (usually 1.0).
            num_rbf (int): Number of RBF centers.
            gamma (float): The width/spread parameter of the Gaussian functions.
        """
        super().__init__()
        self.start = start
        self.stop = stop
        self.num_rbf = num_rbf
        self.gamma = gamma

        # Centers are uniformly spaced over the cosine range [-1, 1]
        centers = torch.linspace(start, stop, num_rbf)
        self.register_buffer("centers", centers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of cosine values. Shape (..., 1) or (...).

        Returns:
            torch.Tensor: Expanded RBF features. Shape (..., num_rbf).
        """
        # Ensure input has a last dimension for broadcasting
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Compute squared difference
        diff = x - self.centers

        # Apply Gaussian function
        return torch.exp(-self.gamma * (diff**2))
