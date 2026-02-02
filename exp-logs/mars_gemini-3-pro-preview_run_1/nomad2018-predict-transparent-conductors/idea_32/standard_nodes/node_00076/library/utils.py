import numpy as np
import os
from library.config import Config


def parse_xyz(file_path):
    """
    Parses an XYZ file to extract lattice vectors, atomic species, and coordinates.

    Args:
        file_path (str): Path to the geometry.xyz file.

    Returns:
        tuple:
            - lattice_vectors (np.ndarray): 3x3 array of lattice vectors.
            - atom_types (list): List of atomic symbols (str).
            - coords (np.ndarray): Nx3 array of atomic Cartesian coordinates.
    """
    lattice_vectors = []
    atom_types = []
    coords = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "lattice_vector":
                # Format: lattice_vector x y z
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                # Format: atom x y z symbol
                coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    # Ensure coords has shape (N, 3) even if N=0
    # Cite debug_lesson_4: Enforce Feature Dimensions on Empty Arrays
    coords_arr = np.array(coords)
    if coords_arr.ndim == 1:
        coords_arr = coords_arr.reshape(-1, 3)

    return np.array(lattice_vectors), atom_types, coords_arr


def compute_pbc_distances(coords, lattice):
    """
    Calculates the pairwise distance matrix between atoms respecting periodic boundary conditions
    using the Minimum Image Convention (MIC). This is used to find the nearest neighbor distance
    for each atom within the unit cell context.

    Args:
        coords (np.ndarray): (N, 3) array of atomic coordinates.
        lattice (np.ndarray): (3, 3) array of lattice vectors.

    Returns:
        np.ndarray: (N, N) array of pairwise distances.
    """
    # Compute inverse lattice to convert to fractional coordinates
    inv_lattice = np.linalg.inv(lattice)
    frac_coords = np.dot(coords, inv_lattice)

    # Compute differences in fractional space: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]

    # Apply Minimum Image Convention: wrap fractional differences to [-0.5, 0.5]
    diff = diff - np.round(diff)

    # Convert back to Cartesian coordinates: D_cart = D_frac @ Lattice
    cart_diff = np.dot(diff, lattice)

    # Compute Euclidean distances
    distances = np.sqrt(np.sum(cart_diff**2, axis=-1))

    return distances


def calculate_idw_chemical_counts(coords, atom_types, lattice, k=Config.K_NEIGHBORS):
    """
    Calculates Inverse-Distance Weighted Chemical Counts (IDW-CC) for each atom.

    This function explicitly handles the "Interaction Blindness" of point clouds by
    aggregating the chemical identity of the local environment. It uses a 3x3x3
    supercell expansion to correctly identify the k-nearest neighbors in a periodic system,
    ensuring that neighbors across boundaries are captured even if they are closer
    than atoms within the primary unit cell.

    Args:
        coords (np.ndarray): (N, 3) array of atomic coordinates.
        atom_types (list): List of N atomic symbols.
        lattice (np.ndarray): (3, 3) array of lattice vectors.
        k (int): Number of nearest neighbors to consider.

    Returns:
        np.ndarray: (N, 4) array where each row is the IDW-CC vector [Al, Ga, In, O].
    """
    N = len(coords)
    # Map atom types to integer indices based on Config
    type_indices = np.array([Config.ATOM_MAP[t] for t in atom_types])

    # Generate shift vectors for a 3x3x3 supercell (indices -1, 0, 1)
    shifts = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                shifts.append([x, y, z])
    shifts = np.array(shifts)  # Shape: (27, 3)

    # Convert integer shifts to Cartesian vectors using the lattice
    shift_vecs = np.dot(shifts, lattice)  # Shape: (27, 3)

    # Construct the supercell coordinates and types
    super_coords_list = []
    super_types_list = []

    for shift in shift_vecs:
        super_coords_list.append(coords + shift)
        super_types_list.append(type_indices)

    # Flatten supercell arrays
    if N > 0:
        # Shape: (27*N, 3)
        super_coords = np.vstack(super_coords_list)
        # Shape: (27*N,)
        super_types = np.hstack(super_types_list)
    else:
        super_coords = np.zeros((0, 3))
        super_types = np.zeros((0,), dtype=int)

    # Initialize output feature array
    idw_features = np.zeros((N, Config.NUM_ATOM_TYPES), dtype=np.float32)

    # Compute features for each atom in the original unit cell
    for i in range(N):
        # Calculate Euclidean distances from atom i to all atoms in the supercell
        diffs = super_coords - coords[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))

        # Filter out the atom itself (distance is effectively 0)
        # We use a small epsilon to handle floating point errors
        mask = dists > 1e-6
        valid_dists = dists[mask]
        valid_types = super_types[mask]

        # Find the indices of the k nearest neighbors
        # argpartition is faster than argsort for finding top k, but we need them sorted
        # if we were doing distance-dependent weighting order, but here we just sum.
        # However, to be safe and deterministic, we sort.
        if len(valid_dists) > k:
            sorted_indices = np.argsort(valid_dists)[:k]
        else:
            sorted_indices = np.arange(len(valid_dists))

        k_dists = valid_dists[sorted_indices]
        k_types = valid_types[sorted_indices]

        # Calculate inverse distance weights
        # Add a small epsilon to denominator for numerical stability (though dists > 1e-6)
        weights = 1.0 / (k_dists + 1e-9)

        # Aggregate weights by atom type
        for t, w in zip(k_types, weights):
            idw_features[i, t] += w

    return idw_features
