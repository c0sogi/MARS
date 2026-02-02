import numpy as np
import torch
from library.config import Config


def get_atomic_one_hot(symbol):
    """
    One-hot encodes an atomic symbol based on the configuration.
    Order: Al, Ga, In, O

    Args:
        symbol (str): Atomic symbol (e.g., 'Al', 'Ga').

    Returns:
        np.ndarray: One-hot encoded vector of shape (4,).
    """
    mapping = {"Al": 0, "Ga": 1, "In": 2, "O": 3}
    one_hot = np.zeros(4, dtype=np.float32)
    if symbol in mapping:
        one_hot[mapping[symbol]] = 1.0
    return one_hot


def inverse_log_transform(y_pred):
    """
    Applies the inverse of the log transform: exp(y) - 1.
    Handles both numpy arrays and torch tensors.

    Args:
        y_pred (np.ndarray or torch.Tensor): Log-transformed predictions.

    Returns:
        np.ndarray or torch.Tensor: Predictions in original scale.
    """
    if isinstance(y_pred, torch.Tensor):
        return torch.exp(y_pred) - 1
    return np.exp(y_pred) - 1


def compute_pbc_distance_matrix(coords, lattice):
    """
    Computes the pairwise distance matrix (N x N) using Minimum Image Convention.

    Args:
        coords: (N, 3) numpy array of atomic coordinates.
        lattice: (3, 3) numpy array of lattice vectors.

    Returns:
        dist_matrix: (N, N) numpy array of MIC distances.
    """
    # Generate translation vectors for 3x3x3 supercell
    ranges = [-1, 0, 1]
    shifts = []
    for i in ranges:
        for j in ranges:
            for k in ranges:
                shifts.append(i * lattice[0] + j * lattice[1] + k * lattice[2])
    shifts = np.array(shifts)  # Shape: (27, 3)

    N = coords.shape[0]
    dist_matrix = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(i + 1, N):
            diff = coords[i] - coords[j]  # Shape: (3,)
            # Apply all 27 shifts to the difference vector
            diffs = diff + shifts  # Shape: (27, 3)
            # Compute distances for all images
            dists = np.linalg.norm(diffs, axis=1)  # Shape: (27,)
            # Find the minimum distance
            min_dist = np.min(dists)

            dist_matrix[i, j] = min_dist
            dist_matrix[j, i] = min_dist

    return dist_matrix


def get_pbc_neighbors(coords, lattice, k):
    """
    Finds the k nearest neighbors for each atom considering Periodic Boundary Conditions.

    Args:
        coords: (N, 3) numpy array of atomic coordinates.
        lattice: (3, 3) numpy array of lattice vectors.
        k: int, number of neighbors to find.

    Returns:
        neighbor_indices: (N, k) int array of neighbor atom indices (0 to N-1).
        neighbor_distances: (N, k) float array of distances to neighbors.
    """
    N = coords.shape[0]

    # Generate translation vectors for 3x3x3 supercell
    ranges = [-1, 0, 1]
    shifts = []
    for i in ranges:
        for j in ranges:
            for k in ranges:
                shifts.append(i * lattice[0] + j * lattice[1] + k * lattice[2])
    shifts = np.array(shifts)  # Shape: (27, 3)

    # Create supercell coordinates and corresponding original indices
    super_coords = []
    super_indices = []

    for shift in shifts:
        super_coords.append(coords + shift)
        super_indices.append(np.arange(N))

    super_coords = np.vstack(super_coords)  # Shape: (27*N, 3)
    super_indices = np.concatenate(super_indices)  # Shape: (27*N,)

    neighbor_indices = np.zeros((N, k), dtype=int)
    neighbor_distances = np.zeros((N, k), dtype=np.float32)

    for i in range(N):
        # Compute distances from atom i to all atoms in the supercell
        diffs = super_coords - coords[i]
        dists = np.linalg.norm(diffs, axis=1)

        # Sort distances to find nearest neighbors
        sorted_args = np.argsort(dists)

        current_indices = []
        current_dists = []

        count = 0
        for arg in sorted_args:
            d = dists[arg]
            # Exclude self-interaction (distance close to 0)
            if d < 1e-4:
                continue

            current_indices.append(super_indices[arg])
            current_dists.append(d)
            count += 1
            if count == k:
                break

        # If for some reason we found fewer than k, pad with last found
        while len(current_indices) < k:
            current_indices.append(current_indices[-1] if current_indices else 0)
            current_dists.append(current_dists[-1] if current_dists else 0.0)

        neighbor_indices[i] = np.array(current_indices)
        neighbor_distances[i] = np.array(current_dists)

    return neighbor_indices, neighbor_distances


def center_coordinates(coords):
    """
    Centers the coordinates relative to their centroid.

    Args:
        coords: (N, 3) numpy array.

    Returns:
        centered_coords: (N, 3) numpy array.
    """
    centroid = np.mean(coords, axis=0)
    return coords - centroid
