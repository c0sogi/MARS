import torch
import torch.nn as nn
from library.config import Config


class GaussianSmearing(nn.Module):
    """
    Gaussian Smearing (RBF expansion) for scalar features.

    This layer expands a scalar input (e.g., distance or cosine of an angle)
    into a vector of Gaussian radial basis functions.

    Formula:
        exp( - (x - mu)^2 / (2 * sigma^2) )

    Where 'mu' are centers linearly spaced between 'start' and 'stop',
    and 'sigma' is determined by the spacing between centers.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        centered: bool = False,
        trainable: bool = False,
    ):
        """
        Args:
            start (float): The minimum value of the input range.
            stop (float): The maximum value of the input range.
            num_gaussians (int): The number of Gaussian basis functions (output dimension).
            centered (bool): If True, centers are fixed at 0 (not typically used for RBF expansion of distances).
            trainable (bool): If True, the centers (mu) and widths (sigma) are learnable parameters.
        """
        super(GaussianSmearing, self).__init__()

        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians

        # Compute the step size (width)
        # We want num_gaussians centers spaced evenly from start to stop
        offset = torch.linspace(start, stop, num_gaussians)

        # The width (sigma) is set to the distance between centers.
        # This ensures a smooth overlap between basis functions.
        # sigma = (stop - start) / (num_gaussians - 1)
        if num_gaussians > 1:
            width = (stop - start) / (num_gaussians - 1)
        else:
            width = 1.0  # Fallback for single gaussian

        widths = torch.FloatTensor([width] * num_gaussians)

        if trainable:
            self.centers = nn.Parameter(offset)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("centers", offset)
            self.register_buffer("widths", widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (..., 1) or (...,).
                              Contains the scalar values to be expanded.

        Returns:
            torch.Tensor: Output tensor of shape (..., num_gaussians).
        """
        # Ensure input has the correct shape for broadcasting
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (N, 1)

        # Calculate the expansion
        # x: (N, 1)
        # centers: (num_gaussians,) -> broadcasts to (1, num_gaussians)
        # diff: (N, num_gaussians)
        diff = x - self.centers

        # Gaussian formula: exp( - (x - mu)^2 / (2 * sigma^2) )
        # Using 2*sigma^2 in denominator is standard for Gaussian distribution.
        # Some implementations use beta * (x - mu)^2 where beta = 1/sigma^2.
        # Here we follow the standard exp(-0.5 * ((x-mu)/sigma)^2)

        z = diff / self.widths
        return torch.exp(-0.5 * (z**2))

    def __repr__(self):
        return f"GaussianSmearing(start={self.start}, stop={self.stop}, num_gaussians={self.num_gaussians})"
