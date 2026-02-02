import os
import random
import numpy as np
import torch
from itertools import product
from library.config import SEED


def set_seed(seed=SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
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


def compute_cell_volume(a, b, c, alpha, beta, gamma):
    """
    Calculates the volume of a unit cell given lattice parameters.

    Args:
        a, b, c (float): Lattice vector lengths.
        alpha, beta, gamma (float): Lattice angles in degrees.

    Returns:
        float: The volume of the unit cell.
    """
    # Convert angles to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Formula for volume of a parallelepiped defined by lengths and angles
    term = (
        1
        - np.cos(alpha_rad) ** 2
        - np.cos(beta_rad) ** 2
        - np.cos(gamma_rad) ** 2
        + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )

    # Ensure non-negative value inside sqrt (numerical stability)
    volume = a * b * c * np.sqrt(max(0.0, term))
    return volume


def cartesian_to_fractional(coords, lattice_vectors):
    """
    Transforms Cartesian coordinates to fractional coordinates.

    Args:
        coords (np.ndarray): Shape (N, 3), Cartesian coordinates.
        lattice_vectors (np.ndarray): Shape (3, 3), Lattice vectors as rows.

    Returns:
        np.ndarray: Shape (N, 3), Fractional coordinates.
    """
    # Fractional = Cartesian * Inverse(Lattice_Matrix)
    # Assuming lattice_vectors rows are a1, a2, a3.
    # r = u*a1 + v*a2 + w*a3
    # r = [u, v, w] @ [a1; a2; a3]
    # [u, v, w] = r @ inv([a1; a2; a3])
    inv_lattice = np.linalg.inv(lattice_vectors)
    fractional = np.dot(coords, inv_lattice)
    return fractional


def get_pbc_distances(coords, lattice_vectors):
    """
    Calculates the distance to the nearest neighbor for each atom,
    respecting Periodic Boundary Conditions (PBC).

    Args:
        coords (np.ndarray): Shape (N, 3), Cartesian coordinates.
        lattice_vectors (np.ndarray): Shape (3, 3), Lattice vectors as rows.

    Returns:
        np.ndarray: Shape (N, 1), Distance to the nearest neighbor for each atom.
    """
    n_atoms = coords.shape[0]

    # If there is only one atom in the unit cell, the nearest neighbor is its own image.
    # The logic below handles this naturally by checking images.

    # Generate 27 shift vectors (including 0,0,0)
    # Indices: -1, 0, 1
    ranges = [-1, 0, 1]
    shifts_indices = np.array(list(product(ranges, repeat=3)))  # (27, 3)
    shifts = np.dot(shifts_indices, lattice_vectors)  # (27, 3)

    # Compute pairwise differences without PBC first: r_i - r_j
    # Shape: (N, N, 3)
    # diffs[i, j, :] = coords[i] - coords[j]
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    # Expand diffs to include shifts: (N, N, 1, 3) - (1, 1, 27, 3) -> (N, N, 27, 3)
    # Effectively calculating: (r_i - r_j) - shift
    # We want min |r_i - r_j - shift| = min |r_i - (r_j + shift)|
    # Note: The sign of shift doesn't matter since we check -1, 0, 1.
    # Let's use addition for clarity: diffs + shifts
    all_diffs = diffs[:, :, np.newaxis, :] + shifts[np.newaxis, np.newaxis, :, :]

    # Compute squared distances: (N, N, 27)
    dists_sq = np.sum(all_diffs**2, axis=-1)

    # We need to mask the self-distance at shift (0,0,0)
    # The shift (0,0,0) is usually at index 13 in the product list if sorted,
    # but let's find it explicitly to be safe.
    zero_shift_idx = np.where((shifts_indices == 0).all(axis=1))[0][0]

    # Mask self-interaction: dists_sq[i, i, zero_shift_idx] = infinity
    # We create a mask of shape (N, N, 27)
    mask = np.ones_like(dists_sq, dtype=bool)

    # Identify diagonal elements (self-pairs)
    diag_indices = np.arange(n_atoms)
    mask[diag_indices, diag_indices, zero_shift_idx] = False

    # Apply mask: set ignored distances to infinity
    dists_sq_masked = np.where(mask, dists_sq, np.inf)

    # Find minimum distance for each atom i
    # Min over j (other atoms) and k (shifts)
    # Flatten last two dimensions: (N, N*27)
    dists_sq_flat = dists_sq_masked.reshape(n_atoms, -1)
    min_dists_sq = np.min(dists_sq_flat, axis=1)

    # Sqrt to get actual distances
    min_dists = np.sqrt(min_dists_sq)

    return min_dists[:, np.newaxis]  # Shape (N, 1)
