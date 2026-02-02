import random
import os
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def rmsle(y_true, y_pred):
    """
    Calculates the Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: The RMSLE score.
    """
    # Clip predictions to be non-negative to avoid log errors
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    squared_log_errors = (np.log1p(y_true) - np.log1p(y_pred)) ** 2
    mean_squared_log_error = np.mean(squared_log_errors)
    return np.sqrt(mean_squared_log_error)


def compute_pbc_distances(coords, lattice):
    """
    Computes pairwise distances and relative vectors respecting Periodic Boundary Conditions (PBC)
    using the Minimum Image Convention.

    Args:
        coords (np.ndarray): Atomic coordinates of shape (N, 3).
        lattice (np.ndarray): Lattice vectors of shape (3, 3).

    Returns:
        tuple:
            - distances (np.ndarray): Pairwise distance matrix of shape (N, N).
            - diff_vectors (np.ndarray): Pairwise relative vectors (i -> j) of shape (N, N, 3).
    """
    # Calculate inverse lattice to convert to fractional coordinates
    # lattice rows are the lattice vectors a1, a2, a3
    inv_lattice = np.linalg.inv(lattice)

    # Convert Cartesian coordinates to fractional coordinates
    # coords (N, 3) @ inv_lattice (3, 3) -> frac_coords (N, 3)
    frac_coords = coords @ inv_lattice

    # Compute pairwise differences in fractional space
    # Shape: (N, N, 3)
    # diff_frac[i, j] = frac[i] - frac[j]
    diff_frac = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]

    # Apply Minimum Image Convention: ensure fractional diffs are in [-0.5, 0.5]
    diff_frac -= np.round(diff_frac)

    # Convert back to Cartesian coordinates
    # diff_cart[i, j] = diff_frac[i, j] @ lattice
    diff_vectors = diff_frac @ lattice

    # Compute Euclidean distances
    distances = np.linalg.norm(diff_vectors, axis=-1)

    return distances, diff_vectors


def compute_inertia_eigenvalues(neighbor_vectors):
    """
    Computes the eigenvalues of the covariance matrix of neighbor positions.
    These eigenvalues serve as rotationally invariant shape descriptors (local inertia).

    Args:
        neighbor_vectors (np.ndarray): Relative vectors to neighbors of shape (K, 3).

    Returns:
        np.ndarray: Sorted eigenvalues (lambda_1, lambda_2, lambda_3) of shape (3,).
                    Returns zeros if K < 2 or input is degenerate.
    """
    # If there are not enough neighbors to form a covariance matrix, return zeros
    if neighbor_vectors.shape[0] < 2:
        return np.zeros(3, dtype=np.float32)

    # Compute covariance matrix (3x3)
    # rowvar=False indicates that rows are observations (neighbors) and cols are variables (x,y,z)
    cov_matrix = np.cov(neighbor_vectors, rowvar=False)

    # Handle edge case where cov_matrix might be scalar or lower dim if input is degenerate
    if cov_matrix.ndim < 2:
        return np.zeros(3, dtype=np.float32)

    # Compute eigenvalues (for symmetric matrix)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)

    # Sort eigenvalues in ascending order (or descending, consistency matters)
    # np.linalg.eigvalsh usually returns them in ascending order
    # We ensure they are sorted to maintain invariance permutation
    eigenvalues.sort()

    return eigenvalues.astype(np.float32)
