import numpy as np
from library.config import COVALENT_RADII, ATOM_TO_IDX


def calculate_angular_distortion(alpha, beta, gamma):
    """
    Computes the angular distortion of the lattice.
    Delta_angle = sum(|theta - 90|) for theta in {alpha, beta, gamma}.

    Args:
        alpha (float): Lattice angle alpha in degrees.
        beta (float): Lattice angle beta in degrees.
        gamma (float): Lattice angle gamma in degrees.

    Returns:
        float: Angular distortion value.
    """
    distortion = abs(alpha - 90.0) + abs(beta - 90.0) + abs(gamma - 90.0)
    return distortion


def get_pbc_neighbors(coords, lattice, atom_types, max_k):
    """
    Finds the k-nearest neighbors for each atom under periodic boundary conditions.

    Args:
        coords (np.ndarray): (N, 3) array of atomic coordinates.
        lattice (np.ndarray): (3, 3) array of lattice vectors.
        atom_types (list): List of atom type strings (length N).
        max_k (int): Number of neighbors to find.

    Returns:
        tuple:
            - neighbor_distances (np.ndarray): (N, max_k) array of distances.
            - neighbor_type_indices (np.ndarray): (N, max_k) array of atom type indices (0-3).
    """
    N = len(coords)

    # Generate supercell offsets (-1, 0, 1)
    # 3x3x3 = 27 cells
    ranges = [-1, 0, 1]
    offsets = np.array(
        [[i, j, k] for i in ranges for j in ranges for k in ranges]
    )  # (27, 3)

    # Calculate cartesian shifts for each image cell
    # v = n1*a1 + n2*a2 + n3*a3
    cart_shifts = offsets @ lattice  # (27, 3)

    # Original types mapped to indices
    type_indices = np.array([ATOM_TO_IDX[t] for t in atom_types])  # (N,)

    # Replicate type indices for supercell
    super_type_indices = np.tile(type_indices, 27)  # (27*N,)

    # Replicate coords for supercell
    # coords: (N, 3)
    # cart_shifts: (27, 3)
    # We want (27*N, 3)
    # Broadcasting: (27, 1, 3) + (1, N, 3) -> (27, N, 3) -> reshape
    super_coords = (cart_shifts[:, np.newaxis, :] + coords[np.newaxis, :, :]).reshape(
        -1, 3
    )

    # Compute distance matrix (N, 27*N)
    # Using broadcasting: (N, 1, 3) - (1, 27*N, 3)
    delta = coords[:, np.newaxis, :] - super_coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(delta**2, axis=2))  # (N, 27*N)

    # Mask self-distances (distance ~ 0 for the image (0,0,0) and same atom index)
    dists[dists < 1e-4] = np.inf

    # Find k nearest neighbors
    num_candidates = dists.shape[1]
    k = min(max_k, num_candidates)

    # Get indices of k smallest values
    # argpartition is faster than sort for large arrays
    if k < num_candidates:
        partitioned_indices = np.argpartition(dists, k, axis=1)[:, :k]
    else:
        partitioned_indices = np.arange(num_candidates)[np.newaxis, :].repeat(N, axis=0)

    # Retrieve values corresponding to these indices
    row_indices = np.arange(N)[:, np.newaxis]
    subset_dists = dists[row_indices, partitioned_indices]

    # Sort the subset to ensure nearest neighbors are ordered
    order = np.argsort(subset_dists, axis=1)

    final_indices = partitioned_indices[row_indices, order]
    final_dists = subset_dists[row_indices, order]

    # Map supercell indices back to type indices
    final_types = super_type_indices[final_indices]

    return final_dists, final_types


def compute_weighted_context(neighbor_dists, neighbor_type_indices, num_types=4):
    """
    Computes inverse-distance weighted chemical context vectors.

    Args:
        neighbor_dists (np.ndarray): (N, K) array of distances.
        neighbor_type_indices (np.ndarray): (N, K) array of type indices.
        num_types (int): Number of atom types (4 for Al, Ga, In, O).

    Returns:
        np.ndarray: (N, num_types) context vectors.
    """
    # Avoid division by zero
    weights = 1.0 / (neighbor_dists + 1e-6)

    # Normalize weights along K axis
    sum_weights = np.sum(weights, axis=1, keepdims=True)
    norm_weights = weights / (sum_weights + 1e-8)  # (N, K)

    # Create one-hot encoding for neighbors: (N, K, num_types)
    N, K = neighbor_type_indices.shape
    one_hot = np.zeros((N, K, num_types))

    # Advanced indexing to set ones
    # We want one_hot[i, j, type_idx] = 1
    i_grid = np.arange(N)[:, np.newaxis]  # (N, 1)
    j_grid = np.arange(K)[np.newaxis, :]  # (1, K)

    one_hot[i_grid, j_grid, neighbor_type_indices] = 1.0

    # Weighted sum: sum_k (weight_ik * one_hot_ik) -> (N, num_types)
    context = np.sum(norm_weights[:, :, np.newaxis] * one_hot, axis=1)

    return context


def calculate_bond_hardness(
    d_min, atom_types, neighbor_type_indices, neighbor_dists, k_context=6
):
    """
    Calculates the Bond Hardness Proxy.
    H = d_min / (r_cov_central + r_avg_neigh)

    Args:
        d_min (np.ndarray): (N,) array of distances to nearest neighbor.
        atom_types (list): List of atom type strings for central atoms.
        neighbor_type_indices (np.ndarray): (N, max_k) array of neighbor type indices.
        neighbor_dists (np.ndarray): (N, max_k) array of neighbor distances.
        k_context (int): Number of neighbors to use for average radius context.

    Returns:
        np.ndarray: (N,) array of bond hardness values.
    """
    N = len(atom_types)

    # Central atom radii
    r_cov_central = np.array([COVALENT_RADII[t] for t in atom_types])

    # Neighbor radii lookup
    radius_lookup = np.zeros(4)
    for atom, radius in COVALENT_RADII.items():
        if atom in ATOM_TO_IDX:
            radius_lookup[ATOM_TO_IDX[atom]] = radius

    # Get radii for neighbors (N, K)
    # Limit to k_context
    k = min(k_context, neighbor_type_indices.shape[1])
    relevant_type_indices = neighbor_type_indices[:, :k]
    relevant_dists = neighbor_dists[:, :k]

    neighbor_radii = radius_lookup[relevant_type_indices]  # (N, k)

    # Calculate weighted average neighbor radius
    weights = 1.0 / (relevant_dists + 1e-6)
    sum_weights = np.sum(weights, axis=1)

    # Weighted sum of radii
    weighted_radii_sum = np.sum(weights * neighbor_radii, axis=1)
    r_avg_neigh = weighted_radii_sum / (sum_weights + 1e-8)

    # Calculate Hardness
    # H = d_min / (r_cov + r_avg)
    bond_hardness = d_min / (r_cov_central + r_avg_neigh + 1e-8)

    return bond_hardness
