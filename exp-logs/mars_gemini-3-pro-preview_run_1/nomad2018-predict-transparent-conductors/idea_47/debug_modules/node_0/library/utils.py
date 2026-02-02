import numpy as np

# Physical constants and properties
# Mass in atomic mass units (u)
# Covalent radius in Angstroms
# Electronegativity (Pauling scale)
ATOMIC_PROPS = {
    "Al": {"mass": 26.981539, "radius": 1.21, "electronegativity": 1.61},
    "Ga": {"mass": 69.723, "radius": 1.22, "electronegativity": 1.81},
    "In": {"mass": 114.818, "radius": 1.42, "electronegativity": 1.78},
    "O": {"mass": 15.999, "radius": 0.66, "electronegativity": 3.44},
}


def calculate_cell_volume(a, b, c, alpha_deg, beta_deg, gamma_deg):
    """
    Calculates the volume of a unit cell given lattice lengths and angles.

    Args:
        a, b, c: Lattice vector lengths (float).
        alpha_deg, beta_deg, gamma_deg: Lattice angles in degrees (float).

    Returns:
        volume: Unit cell volume (float).
    """
    alpha = np.radians(alpha_deg)
    beta = np.radians(beta_deg)
    gamma = np.radians(gamma_deg)

    term = (
        1
        - np.cos(alpha) ** 2
        - np.cos(beta) ** 2
        - np.cos(gamma) ** 2
        + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
    )

    # Ensure non-negative under sqrt due to floating point errors
    volume = a * b * c * np.sqrt(np.maximum(0, term))
    return volume


def calculate_angular_distortion(alpha_deg, beta_deg, gamma_deg):
    """
    Calculates the angular distortion of the unit cell (deviation from 90 degrees).

    Args:
        alpha_deg, beta_deg, gamma_deg: Lattice angles in degrees (float).

    Returns:
        distortion: Sum of absolute deviations from 90 degrees (float).
    """
    distortion = (
        np.abs(alpha_deg - 90.0) + np.abs(beta_deg - 90.0) + np.abs(gamma_deg - 90.0)
    )
    return distortion


def get_pbc_distances(positions, lattice, k_neighbors):
    """
    Computes distances and indices of k-nearest neighbors under periodic boundary conditions.

    Args:
        positions: Atomic coordinates (N, 3).
        lattice: Lattice vectors (3, 3).
        k_neighbors: Number of neighbors to find.

    Returns:
        distances: (N, k) array of distances to nearest neighbors.
        indices: (N, k) array of indices of nearest neighbors.
    """
    n_atoms = len(positions)

    # Generate image offsets (3x3x3 grid = 27 images)
    # We use a range of -1 to 1 for periodic images which is usually sufficient
    # for finding nearest neighbors in these types of dense crystal structures.

    # Create grid of indices [-1, 0, 1]
    x, y, z = np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")
    # Stack to get (27, 3) array of multipliers
    multipliers = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Calculate translation vectors: (27, 3)
    translations = multipliers @ lattice

    # Create supercell positions: (27 * N, 3)
    # We repeat positions for each translation
    # shape: (27, N, 3) -> reshape to (27*N, 3)
    supercell_pos = (positions[None, :, :] + translations[:, None, :]).reshape(-1, 3)

    # Create supercell indices mapping back to original atoms: (27 * N,)
    supercell_indices = np.tile(np.arange(n_atoms), 27)

    # Compute pairwise distances
    # Distance matrix: (N, 27*N)
    # Using broadcasting: (N, 1, 3) - (1, 27*N, 3) -> (N, 27*N, 3) -> norm -> (N, 27*N)

    diff = positions[:, None, :] - supercell_pos[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))

    # Mask self distances (only strictly 0 or epsilon for numerical stability)
    # We set small distances (self) to inf to push them to the end during sorting
    dists[dists < 1e-5] = np.inf

    # Ensure we don't try to find more neighbors than available points
    k = min(k_neighbors, dists.shape[1])

    # Get indices of k smallest elements using argpartition for efficiency
    partitioned_indices = np.argpartition(dists, k, axis=1)[:, :k]

    # Gather the distances corresponding to these indices
    row_indices = np.arange(n_atoms)[:, None]
    top_k_dists = dists[row_indices, partitioned_indices]

    # Sort within the top k to ensure ordered output (nearest first)
    sort_order = np.argsort(top_k_dists, axis=1)

    # Apply sort order to distances and indices
    sorted_dists = top_k_dists[row_indices, sort_order]
    sorted_supercell_indices = partitioned_indices[row_indices, sort_order]

    # Map supercell indices back to original atom indices (0 to N-1)
    final_indices = supercell_indices[sorted_supercell_indices]

    return sorted_dists, final_indices
