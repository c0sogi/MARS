import torch
import torch.nn as nn
import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import brentq


class Envelope(nn.Module):
    """
    Polynomial envelope function ensuring smooth cutoff.
    u(x) = 1 - 6x^5 + 15x^4 - 10x^3 for x in [0, 1]
    """

    def __init__(self, exponent: int = 5):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        # x is normalized distance d / cutoff
        # Enforce x in [0, 1] implicitly by caller or clamp if needed,
        # but usually handled by mask in GNNs.
        # Here we compute the polynomial.
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x

        # The DimeNet envelope formula:
        return 1.0 / x * (a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) + (
            1.0 if p == 6 else 0.0
        )


class RadialBasisFunctions(nn.Module):
    """
    Bessel Basis Functions for distance encoding.
    Features are proportional to sin(n * pi * d / cutoff) / d.
    """

    def __init__(self, num_radial: int, cutoff: float, envelope_exponent: int = 5):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.inv_cutoff = 1.0 / cutoff

        # Initialize frequencies: n * pi
        self.frequencies = nn.Parameter(
            torch.Tensor(np.pi * np.arange(1, num_radial + 1)), requires_grad=False
        )
        self.envelope = Envelope(envelope_exponent)

    def forward(self, d):
        """
        Args:
            d: Tensor of distances of shape [...]
        Returns:
            Tensor of shape [..., num_radial]
        """
        d = d.unsqueeze(-1) if d.dim() == 1 else d
        d_scaled = d * self.inv_cutoff

        env = self.envelope(d_scaled)

        # Avoid division by zero for very small distances
        d_safe = torch.clamp(d, min=1e-7)

        return env * (torch.sin(self.frequencies * d_scaled) / d_safe)


class SphericalBasisFunctions(nn.Module):
    """
    Spherical Basis Functions for triplet (distance, angle) encoding.
    Combines Spherical Bessel functions (radial) and Legendre Polynomials (angular).
    """

    def __init__(
        self,
        num_radial: int,
        num_spherical: int,
        cutoff: float,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        # Precompute roots of Spherical Bessel functions
        # Roots of j_l(z) are roots of J_{l+0.5}(z)
        # We compute them numerically using scanning + Brent's method
        roots = []
        for l in range(num_spherical):
            l_roots = []

            # Scanning parameters
            # Roots are roughly pi apart. A step of 0.1 is safe.
            step = 0.1
            x = step

            # Robust scanning for roots
            while len(l_roots) < num_radial:
                y1 = spherical_jn(l, x)
                y2 = spherical_jn(l, x + step)

                if y1 * y2 < 0:
                    # Sign change detected, root is in [x, x+step]
                    r = brentq(lambda z: spherical_jn(l, z), x, x + step)
                    l_roots.append(r)
                elif y1 == 0:
                    # Exact zero found (unlikely with float, but possible)
                    l_roots.append(x)

                x += step

                # Safety break (roots are approx n*pi)
                if x > (num_radial + l + 10) * np.pi:
                    # Fallback if we can't find enough roots (should not happen)
                    break

            # If we missed some roots (rare), fill with approximation
            while len(l_roots) < num_radial:
                n = len(l_roots) + 1
                l_roots.append((n + l / 2.0) * np.pi)

            roots.append(l_roots[:num_radial])

        # Shape: [num_spherical, num_radial]
        self.register_buffer("roots", torch.from_numpy(np.array(roots)).float())

    def forward(self, d, theta):
        """
        Args:
            d: Tensor of distances [N]
            theta: Tensor of angles [N] (radians)
        Returns:
            Tensor of shape [N, num_spherical * num_radial]
        """
        # 1. Envelope
        d_scaled = d * (1.0 / self.cutoff)
        env = self.envelope(d_scaled)  # [N]

        # 2. Radial Part: Spherical Bessel Functions j_l(z_{ln} * d / c)
        # We compute j_l recursively for differentiability

        # Arguments for the Bessel functions: x = z_{ln} * (d / c)
        # self.roots shape: [L, n_rad]
        # d_scaled shape: [N]
        # x shape: [N, L, n_rad]
        x = d_scaled.unsqueeze(-1).unsqueeze(-1) * self.roots.unsqueeze(0)

        # Compute j_l(x) recursively
        # j_0(x) = sin(x)/x
        # j_1(x) = sin(x)/x^2 - cos(x)/x

        # Avoid division by zero
        x_safe = torch.clamp(x, min=1e-7)
        sin_x = torch.sin(x_safe)
        cos_x = torch.cos(x_safe)

        # Initialize list to hold j_l values for l=0..L-1
        bessel_funcs = []

        # l = 0
        if self.num_spherical > 0:
            j0 = sin_x / x_safe
            bessel_funcs.append(j0)

        # l = 1
        if self.num_spherical > 1:
            j1 = (sin_x / x_safe.pow(2)) - (cos_x / x_safe)
            bessel_funcs.append(j1)

        # l > 1: Recurrence j_{l+1} = (2l+1)/x * j_l - j_{l-1}
        for l in range(1, self.num_spherical - 1):
            j_prev = bessel_funcs[-2]
            j_curr = bessel_funcs[-1]
            j_next = ((2 * l + 1) / x_safe) * j_curr - j_prev
            bessel_funcs.append(j_next)

        # Stack radial parts: [N, num_spherical, num_radial]
        # We only want the diagonal terms where the Bessel function order matches the root set order
        radial_list = []
        for l, bf in enumerate(bessel_funcs):
            # bf has shape [N, L, n_rad], we want the l-th slice of dim 1
            radial_list.append(bf[:, l, :])
        radial = torch.stack(radial_list, dim=1)

        # 3. Angular Part: Legendre Polynomials P_l(cos theta)
        # Computed recursively
        cos_theta = torch.cos(theta)  # [N]

        legendre_funcs = []

        # l = 0: P_0(x) = 1
        if self.num_spherical > 0:
            p0 = torch.ones_like(cos_theta)
            legendre_funcs.append(p0)

        # l = 1: P_1(x) = x
        if self.num_spherical > 1:
            p1 = cos_theta
            legendre_funcs.append(p1)

        # l > 1: (l+1) P_{l+1} = (2l+1) x P_l - l P_{l-1}
        # => P_{l} = ((2l-1) x P_{l-1} - (l-1) P_{l-2}) / l
        for l in range(2, self.num_spherical):
            p_prev = legendre_funcs[-2]
            p_curr = legendre_funcs[-1]
            p_next = ((2 * l - 1) * cos_theta * p_curr - (l - 1) * p_prev) / l
            legendre_funcs.append(p_next)

        # Stack angular parts: [N, num_spherical]
        angular = torch.stack(legendre_funcs, dim=1)

        # 4. Combine
        # Output shape should be [N, num_spherical * num_radial]
        # We multiply radial [N, L, n_rad] by angular [N, L] (broadcasted)
        # and then by envelope [N]

        # Expand angular to [N, L, 1]
        angular = angular.unsqueeze(-1)

        # Product
        sbf = radial * angular  # [N, L, n_rad]

        # Flatten L and n_rad
        sbf = sbf.view(sbf.shape[0], -1)  # [N, L * n_rad]

        # Apply envelope
        return env.unsqueeze(-1) * sbf
