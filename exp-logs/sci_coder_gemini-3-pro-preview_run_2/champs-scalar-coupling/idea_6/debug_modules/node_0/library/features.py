import torch
import math
import numpy as np


def polynomial_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    """
    Computes a polynomial cutoff envelope function.
    u(d) = 1 - 6(d/c)^5 + 15(d/c)^4 - 10(d/c)^3  for d < c
    u(d) = 0                                     for d >= c

    Args:
        distances (torch.Tensor): Tensor of inter-atomic distances.
        cutoff (float): The cutoff distance.

    Returns:
        torch.Tensor: Envelope values in range [0, 1].
    """
    # Ensure distances are non-negative
    d = distances.clamp(min=0.0)

    # Calculate scaled distance x = d / cutoff
    x = d / cutoff

    # Compute polynomial: 1 - 6x^5 + 15x^4 - 10x^3
    # Factorized: 1 - x^3 * (10 - 15x + 6x^2)
    envelope = 1.0 - (x**3) * (10.0 - 15.0 * x + 6.0 * (x**2))

    # Apply zero mask for d >= cutoff
    mask = (d < cutoff).float()

    return envelope * mask


def radial_basis_functions(
    distances: torch.Tensor,
    start: float = 0.0,
    end: float = 5.0,
    num_basis: int = 50,
    sigma: float = None,
) -> torch.Tensor:
    """
    Expands distances using Gaussian Radial Basis Functions (RBF) with a polynomial envelope.

    Expansion: e_k(d) = exp(- (d - mu_k)^2 / (2 * sigma^2)) * envelope(d)

    Args:
        distances (torch.Tensor): Tensor of distances (shape: [N] or [N, 1]).
        start (float): Minimum center for RBF.
        end (float): Maximum center for RBF (and cutoff value).
        num_basis (int): Number of Gaussian centers.
        sigma (float, optional): Width of Gaussians. If None, calculated from spacing.

    Returns:
        torch.Tensor: RBF features (shape: [N, num_basis]).
    """
    if not isinstance(distances, torch.Tensor):
        distances = torch.tensor(distances, dtype=torch.float32)

    # Ensure shape [N, 1] for broadcasting
    if distances.dim() == 1:
        d = distances.unsqueeze(-1)
    else:
        d = distances

    # Setup centers (mu)
    centers = torch.linspace(start, end, num_basis, device=d.device, dtype=d.dtype)

    # Setup width (sigma)
    if sigma is None:
        # Default sigma is the distance between centers
        step = (end - start) / num_basis
        sigma = step

    # Compute Gaussian expansion
    # (d - mu)^2
    diff = d - centers  # Broadcasting: [N, 1] - [num_basis] -> [N, num_basis]
    rbf = torch.exp(-(diff**2) / (2.0 * sigma**2))

    # Apply envelope
    envelope = polynomial_cutoff(d, end)

    return rbf * envelope


def spherical_basis_functions(
    distances: torch.Tensor,
    angles: torch.Tensor,
    start: float = 0.0,
    end: float = 5.0,
    num_basis: int = 50,
) -> torch.Tensor:
    """
    Expands geometric triplets (distance, angle) using a 2D Spherical Basis.
    Constructed as the outer product of a Radial Basis and an Angular Basis.

    SBF(d, theta) = RBF(d) (outer) Angular(theta)

    Args:
        distances (torch.Tensor): Tensor of distances (shape: [N]).
        angles (torch.Tensor): Tensor of angles in radians (shape: [N]).
        start (float): Minimum center for radial component.
        end (float): Maximum center/cutoff for radial component.
        num_basis (int): Total number of output features.
                         Will be split into num_rad x num_ang.

    Returns:
        torch.Tensor: SBF features (shape: [N, num_basis]).
    """
    if not isinstance(distances, torch.Tensor):
        distances = torch.tensor(distances, dtype=torch.float32)
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor(angles, dtype=torch.float32)

    # Determine split for radial and angular basis
    # We want n_rad * n_ang approx num_basis.
    # Heuristic: n_ang is usually smaller than n_rad, but for SBF they are often balanced.
    # Let's try to find factors or just fix a ratio.
    # Common config: 50 -> 5 radial * 10 angular is reasonable? Or 7 * 7 = 49.
    # Let's prioritize angular resolution slightly less than radial?
    # Actually, for 50, 5 radial * 10 angular is a clean split.

    if num_basis >= 50:
        n_rad = 5
        n_ang = num_basis // n_rad
    elif num_basis >= 20:
        n_rad = 4
        n_ang = num_basis // n_rad
    else:
        n_rad = int(math.sqrt(num_basis))
        n_ang = num_basis // n_rad

    # Adjust to ensure exact size match if needed, but outer product is strict.
    # We will compute n_rad * n_ang features. If < num_basis, we pad. If > we slice (unlikely with integer div).
    # To be safe, we'll slice or pad the final output to exactly num_basis.

    # 1. Radial Part (Gaussian RBF)
    # Shape: [N, n_rad]
    rad_feat = radial_basis_functions(distances, start, end, n_rad)

    # 2. Angular Part (Cosine Expansion)
    # Basis: cos(k * theta) for k in 0..n_ang-1
    # Shape: [N, n_ang]
    if angles.dim() == 1:
        theta = angles.unsqueeze(-1)
    else:
        theta = angles

    freqs = torch.arange(0, n_ang, device=theta.device, dtype=theta.dtype)
    ang_feat = torch.cos(theta * freqs)

    # 3. Outer Product
    # We want for each sample i: rad_feat[i] (vector) (outer) ang_feat[i] (vector) -> matrix -> flatten
    # Einstein summation: "ni, nj -> nij"
    # rad_feat: [N, n_rad], ang_feat: [N, n_ang]
    # Result: [N, n_rad, n_ang]
    sbf_matrix = torch.einsum("ni,nj->nij", rad_feat, ang_feat)

    # Flatten to [N, n_rad * n_ang]
    sbf = sbf_matrix.reshape(distances.shape[0], -1)

    # 4. Resize to exactly num_basis
    current_dim = sbf.shape[1]
    if current_dim < num_basis:
        # Pad with zeros
        padding = torch.zeros(
            (sbf.shape[0], num_basis - current_dim), device=sbf.device
        )
        sbf = torch.cat([sbf, padding], dim=1)
    elif current_dim > num_basis:
        # Slice
        sbf = sbf[:, :num_basis]

    return sbf
