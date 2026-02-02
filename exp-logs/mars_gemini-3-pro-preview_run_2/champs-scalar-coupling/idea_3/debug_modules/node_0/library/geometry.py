import torch
import torch.nn as nn
import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import brentq
from library.config import DEVICE, CUTOFF, ENVELOPE_EXPONENT


class Envelope(nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff at the boundary.
    Based on the DimeNet envelope: u(x) = 1 - (p+1)(p+2)/2 * x^p + p(p+2) * x^(p+1) - p(p+1)/2 * x^(p+2)
    where x = d / cutoff.
    """

    def __init__(self, exponent: int = ENVELOPE_EXPONENT):
        super(Envelope, self).__init__()
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of scaled distances (d / cutoff).
        Returns:
            Tensor of envelope values.
        """
        # Ensure x is within [0, 1] for the formula, though we mask > 1 later
        p = self.p
        env_val = 1.0 + self.a * x**p + self.b * x ** (p + 1) + self.c * x ** (p + 2)

        # Apply zero masking for distances greater than cutoff (x > 1)
        return torch.where(x < 1.0, env_val, torch.zeros_like(x))


class RadialBasisFunctions(nn.Module):
    """
    Expands distances into a vector using Gaussian Radial Basis Functions.
    """

    def __init__(self, num_rbf: int, cutoff: float = CUTOFF, start: float = 0.0):
        super(RadialBasisFunctions, self).__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff

        # Compute centers and widths
        offset = torch.linspace(start, cutoff, num_rbf)
        # Width chosen such that functions overlap reasonably
        width = torch.tensor((cutoff - start) / num_rbf)

        self.register_buffer("offset", offset)
        self.register_buffer("width", width)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dist: Tensor of distances (N,).
        Returns:
            Tensor of RBF expansions (N, num_rbf).
        """
        # (N, 1) - (1, num_rbf) -> (N, num_rbf)
        diff = dist.unsqueeze(-1) - self.offset
        return torch.exp(-1.0 * (diff.pow(2)) / self.width)


class SphericalBasisFunctions(nn.Module):
    """
    Expands triplets (distance, angle) using Spherical Basis Functions.
    Combines Spherical Bessel functions (radial) and Legendre polynomials (angular).
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = CUTOFF,
        envelope_exponent: int = ENVELOPE_EXPONENT,
    ):
        super(SphericalBasisFunctions, self).__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff

        self.envelope = Envelope(envelope_exponent)

        # 1. Precompute Roots of Spherical Bessel Functions
        # We need z_ln such that j_l(z_ln) = 0
        roots = self._compute_bessel_roots(num_spherical, num_radial)
        self.register_buffer("roots", torch.from_numpy(roots).float())

        # 2. Precompute Normalization Constants
        # Pre-calculate normalization to ensure basis is orthonormal within the envelope
        # Note: This is an approximation or standard normalization for Bessel basis in sphere
        # Norm factor: sqrt(2 / (cutoff^3 * j_{l+1}^2(z_ln)))
        # We compute this numerically
        # We need j_{l+1}(z_ln)
        # roots shape: (num_spherical, num_radial)

        # Create a grid of l indices matching roots shape
        l_indices = np.arange(num_spherical)[:, None]  # (L, 1)

        # Calculate j_{l+1}(root)
        # We iterate to apply spherical_jn correctly
        norm_factors = np.zeros_like(roots)
        for l in range(num_spherical):
            for n in range(num_radial):
                root = roots[l, n]
                val = spherical_jn(l + 1, root)
                norm_factors[l, n] = np.sqrt(2.0 / (cutoff**3 * val**2))

        self.register_buffer("norm_factors", torch.from_numpy(norm_factors).float())

    def _compute_bessel_roots(self, num_spherical, num_radial):
        """
        Numerically compute roots of spherical Bessel functions.
        """
        roots = []
        for l in range(num_spherical):
            l_roots = []
            for n in range(1, num_radial + 1):
                # Roots of j_l are approximately at (n + l/2)*pi
                # We search in a bracket around this approximation
                guess = (n + l / 2.0) * np.pi

                # Search window
                a, b = guess - 1.0, guess + 1.0

                # Ensure signs are opposite for brentq
                # If guess is very close to root, expanding window might be needed
                # but usually this approximation is quite good for Bessel
                try:
                    root = brentq(lambda z: spherical_jn(l, z), a, b)
                except ValueError:
                    # Fallback: widen search if approximation was slightly off
                    # or if n=1 and l is small
                    a, b = guess - 2.0, guess + 2.0
                    # Ensure a > 0
                    a = max(1e-4, a)
                    root = brentq(lambda z: spherical_jn(l, z), a, b)

                l_roots.append(root)
            roots.append(l_roots)
        return np.array(roots)

    def forward(
        self, dist: torch.Tensor, angle: torch.Tensor, idx_kj: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dist: Distance from neighbor k to center j. Shape (num_edges_k,).
            angle: Angle at center j (k-j-i). Shape (num_triplets,).
            idx_kj: Index of the edge kj corresponding to each triplet. Shape (num_triplets,).
                    Used to gather radial features for the triplets.

        Returns:
            Tensor of spherical basis features. Shape (num_triplets, num_spherical * num_radial).
        """
        # 1. Radial Part (Spherical Bessel)
        # Gather distances for the triplets
        d = dist[idx_kj]  # (num_triplets,)

        # Scale distance: x = d / cutoff
        x = d / self.cutoff

        # Calculate Bessel functions: j_l(z_ln * x)
        # We need to compute this for all l, n.
        # roots shape: (L, N)
        # x shape: (T,)
        # result shape: (T, L, N)

        # Expand dims for broadcasting
        # x: (T, 1, 1)
        # roots: (1, L, N)
        x_expanded = x.unsqueeze(-1).unsqueeze(-1)
        roots_expanded = self.roots.unsqueeze(0)

        # Argument for sin/cos in Bessel
        args = roots_expanded * x_expanded  # (T, L, N)

        # Implement Spherical Bessel j_l(z) explicitly for differentiability
        # j_l(z) = sqrt(pi/2z) * J_{l+0.5}(z)
        # However, writing generic j_l in pytorch is hard.
        # DimeNet approach: Use the explicit formula or recursion?
        # Standard approach in GNNs (e.g. DimeNet implementation in PyG):
        # They usually implement explicit formulas for small l or use recursion.
        # Given generic num_spherical, recursion is best.

        # Recursive calculation of j_l(z)
        # j_0(z) = sin(z)/z
        # j_1(z) = sin(z)/z^2 - cos(z)/z
        # j_{l}(z) = (2l-1)/z * j_{l-1}(z) - j_{l-2}(z)

        # We compute for max L
        # Flatten args for simpler computation: (T*L*N)
        z = args

        # To avoid division by zero at z=0, we clamp.
        # Though d is usually > 0 in molecules.
        z = torch.clamp(z, min=1e-5)

        sin_z = torch.sin(z)
        cos_z = torch.cos(z)

        # Storage for all orders
        # We need a list of tensors
        bessel_funcs = [None] * self.num_spherical

        # j_0
        bessel_funcs[0] = sin_z / z

        if self.num_spherical > 1:
            # j_1
            bessel_funcs[1] = sin_z / (z**2) - cos_z / z

        for l in range(2, self.num_spherical):
            bessel_funcs[l] = ((2 * l - 1) / z) * bessel_funcs[l - 1] - bessel_funcs[
                l - 2
            ]

        # Stack back: (T, L, N)
        # bessel_list is list of (T, L, N) tensors? No, z was (T, L, N).
        # So bessel_funcs[l] is (T, L, N).
        # But we only need the l-th slice from the l-th calculation?
        # No, z depends on l because roots depends on l.
        # So bessel_funcs[l] calculated on z (which contains all l roots) is wasteful but correct if we slice.
        # Actually, z = roots[l] * x. So z is different for each l.
        # Let's optimize. We iterate l.

        rbf = []
        for l in range(self.num_spherical):
            # Roots for this l: (1, N)
            roots_l = self.roots[l].unsqueeze(0)  # (1, N)
            z_l = x.unsqueeze(-1) * roots_l  # (T, N)
            z_l = torch.clamp(z_l, min=1e-5)

            # Compute j_l(z_l)
            # We need recursion up to l for value z_l
            if l == 0:
                val = torch.sin(z_l) / z_l
            elif l == 1:
                val = torch.sin(z_l) / (z_l**2) - torch.cos(z_l) / z_l
            else:
                # We need to run recursion for specific z_l
                j_prev2 = torch.sin(z_l) / z_l  # j0
                j_prev1 = torch.sin(z_l) / (z_l**2) - torch.cos(z_l) / z_l  # j1
                val = j_prev1  # Placeholder

                for k in range(2, l + 1):
                    val = ((2 * k - 1) / z_l) * j_prev1 - j_prev2
                    j_prev2 = j_prev1
                    j_prev1 = val

            rbf.append(val)

        # Stack: (T, L, N)
        rbf = torch.stack(rbf, dim=1)

        # Apply normalization
        # norm_factors: (L, N) -> (1, L, N)
        rbf = rbf * self.norm_factors.unsqueeze(0)

        # Apply Envelope
        # env: (T,) -> (T, 1, 1)
        u_x = self.envelope(x).unsqueeze(-1).unsqueeze(-1)
        rbf = rbf * u_x

        # 2. Angular Part (Legendre Polynomials)
        # P_l(cos theta)
        # angle shape: (T,)
        cos_theta = torch.cos(angle)

        # Recursion for Legendre
        # P_0(x) = 1
        # P_1(x) = x
        # P_l(x) = ((2l-1)x P_{l-1} - (l-1)P_{l-2}) / l

        legendre = [None] * self.num_spherical
        legendre[0] = torch.ones_like(cos_theta)
        if self.num_spherical > 1:
            legendre[1] = cos_theta

        for l in range(2, self.num_spherical):
            term1 = (2 * l - 1) * cos_theta * legendre[l - 1]
            term2 = (l - 1) * legendre[l - 2]
            legendre[l] = (term1 - term2) / l

        # Stack: (T, L)
        legendre = torch.stack(legendre, dim=1)

        # 3. Combine
        # rbf: (T, L, N)
        # legendre: (T, L) -> (T, L, 1)
        # Output: (T, L, N) -> flatten to (T, L*N)

        out = rbf * legendre.unsqueeze(-1)
        out = out.view(out.size(0), -1)

        return out
