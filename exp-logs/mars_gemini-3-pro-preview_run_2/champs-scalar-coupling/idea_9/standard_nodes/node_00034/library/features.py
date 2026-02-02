import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class RadialBasisFunctions(nn.Module):
    """
    Expands inter-atomic distances using Bessel Basis functions with a polynomial envelope.
    This ensures the representation is continuous and differentiable at the cutoff.

    Formula:
        RBF_n(d) = sqrt(2/c) * sin(n * pi * d / c) / d * Envelope(d/c)
    """

    def __init__(self, cutoff: float = Config.CUTOFF, num_rbf: int = Config.NUM_RBF):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf

        # Precompute frequencies: n * pi for n = 1...num_rbf
        # Registered as buffer so it's part of state_dict but not a learnable parameter
        freqs = torch.arange(1, num_rbf + 1, dtype=torch.float32) * np.pi
        self.register_buffer("freqs", freqs)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d (torch.Tensor): Tensor of distances (shape: ...)

        Returns:
            torch.Tensor: RBF expansion (shape: ..., num_rbf)
        """
        # Ensure d is at least 1D
        d_unsqueeze = d.unsqueeze(-1)  # (..., 1)

        # 1. Scale distance by cutoff
        d_scaled = d_unsqueeze / self.cutoff

        # 2. Polynomial Envelope: 1 - 6x^5 + 15x^4 - 10x^3
        # This ensures smooth decay to 0 at cutoff
        env = (
            1.0
            - 6.0 * torch.pow(d_scaled, 5)
            + 15.0 * torch.pow(d_scaled, 4)
            - 10.0 * torch.pow(d_scaled, 3)
        )

        # Apply envelope only within cutoff (though typically d < cutoff by construction)
        # Using torch.where ensures 0 gradient outside cutoff
        env = torch.where(d_scaled < 1.0, env, torch.zeros_like(d_scaled))

        # 3. Bessel Basis
        # Argument for sin: n * pi * d / c
        args = d_scaled * self.freqs  # (..., num_rbf)

        # sin(args)
        sin_term = torch.sin(args)

        # Normalize: sqrt(2/c) / d
        # Clamp d to avoid division by zero (though physical d > 0.5A usually)
        d_safe = torch.clamp(d_unsqueeze, min=1e-7)
        prefactor = np.sqrt(2.0 / self.cutoff)

        rbf = prefactor * (sin_term / d_safe) * env

        return rbf


class SphericalBasisFunctions(nn.Module):
    """
    Expands geometric triplets (distance d, angle theta) using a tensor product of
    Radial Basis Functions and Legendre Polynomials.

    This provides a rotationally invariant representation of the local angle geometry.
    """

    def __init__(
        self,
        cutoff: float = Config.CUTOFF,
        num_rbf: int = Config.NUM_RBF,
        num_sbf: int = Config.NUM_SBF,
    ):
        super().__init__()
        self.num_sbf = num_sbf
        self.num_rbf = num_rbf

        # Radial part uses the same Bessel expansion logic
        self.rbf = RadialBasisFunctions(cutoff, num_rbf)

    def forward(self, d: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d (torch.Tensor): Distances (shape: ...)
            angle (torch.Tensor): Angles in radians (shape: ...)

        Returns:
            torch.Tensor: SBF expansion (shape: ..., num_rbf * num_sbf)
        """
        # 1. Compute Radial Features
        rbf_features = self.rbf(d)  # (..., num_rbf)

        # 2. Compute Angular Features using Legendre Polynomials P_l(cos theta)
        x = torch.cos(angle)  # (..., )

        # We compute P_l(x) recursively
        # P_0(x) = 1
        # P_1(x) = x
        # P_l(x) = ((2l-1)x * P_{l-1} - (l-1) * P_{l-2}) / l

        legendre_polys = []

        # l=0
        legendre_polys.append(torch.ones_like(x))

        if self.num_sbf > 1:
            # l=1
            legendre_polys.append(x)

        for l in range(2, self.num_sbf):
            p_prev = legendre_polys[-1]
            p_prev2 = legendre_polys[-2]

            # Recurrence relation
            term1 = (2 * l - 1) * x * p_prev
            term2 = (l - 1) * p_prev2
            p_curr = (term1 - term2) / l

            legendre_polys.append(p_curr)

        # Stack angular features
        angular_features = torch.stack(legendre_polys, dim=-1)  # (..., num_sbf)

        # 3. Tensor Product
        # We want every combination of radial basis n and angular basis l
        # Output dim: num_rbf * num_sbf

        # Expand dims for broadcasting: (..., num_rbf, 1) * (..., 1, num_sbf)
        out = rbf_features.unsqueeze(-1) * angular_features.unsqueeze(
            -2
        )  # (..., num_rbf, num_sbf)

        # Flatten the last two dimensions
        out = out.view(*out.shape[:-2], -1)  # (..., num_rbf * num_sbf)

        return out
