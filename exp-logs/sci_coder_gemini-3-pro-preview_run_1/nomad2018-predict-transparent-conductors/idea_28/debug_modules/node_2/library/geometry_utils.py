import numpy as np
import os
from library.config import Config


def parse_xyz(file_path):
    """
    Parses a geometry.xyz file to extract lattice vectors, atomic types, and coordinates.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        tuple:
            - lattice_vectors (np.ndarray): 3x3 array of lattice vectors.
            - atom_types (list): List of atomic species strings (e.g., ['Al', 'Ga', ...]).
            - atom_coords (np.ndarray): Nx3 array of Cartesian atomic coordinates.
    """
    lattice_vectors = []
    atom_types = []
    atom_coords = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == "lattice_vector":
            lattice_vectors.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            # Format: atom x y z type
            atom_coords.append([float(x) for x in parts[1:4]])
            atom_types.append(parts[4])

    return np.array(lattice_vectors), atom_types, np.array(atom_coords)


def compute_pbc_distance_matrix(coords, lattice_vectors):
    """
    Computes the pairwise Euclidean distance matrix respecting Periodic Boundary Conditions (PBC)
    using the Minimum Image Convention.

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.

    Returns:
        np.ndarray: NxN distance matrix.
    """
    # Calculate inverse lattice for fractional coordinate conversion
    inv_lattice = np.linalg.inv(lattice_vectors)

    # Convert to fractional coordinates
    frac_coords = coords @ inv_lattice

    # Compute pairwise differences in fractional space
    # shape: (N, N, 3)
    diff_frac = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]

    # Apply Minimum Image Convention: wrap fractional differences to [-0.5, 0.5]
    diff_frac -= np.round(diff_frac)

    # Convert back to Cartesian space
    diff_cart = diff_frac @ lattice_vectors

    # Compute Euclidean distances
    dist_matrix = np.sqrt(np.sum(diff_cart**2, axis=-1))

    return dist_matrix


def calculate_atomic_features(atom_types, coords, lattice_vectors):
    """
    Calculates node-level features for each atom.

    Features (12 dims):
    1. Atomic Identity (One-hot, 4 dims): Al, Ga, In, O
    2. Spatial Context (3 dims): Centered Cartesian coordinates
    3. Chemically-Resolved Reciprocal Proximity (4 dims): 1/d to nearest neighbor of each type
    4. Local Packing Density (1 dim): Mean distance to K nearest neighbors

    Args:
        atom_types (list): List of atomic species.
        coords (np.ndarray): Nx3 array of atomic coordinates.
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.

    Returns:
        np.ndarray: Nx12 feature matrix.
    """
    num_atoms = len(atom_types)

    # 1. Atomic Identity (One-hot)
    # Map types to indices: Al:0, Ga:1, In:2, O:3
    type_map = {t: i for i, t in enumerate(Config.ATOM_TYPES)}
    one_hot = np.zeros((num_atoms, Config.NUM_ATOM_TYPES))
    for i, t in enumerate(atom_types):
        if t in type_map:
            one_hot[i, type_map[t]] = 1.0

    # 2. Spatial Context (Centered Coordinates)
    # Subtract centroid to make translation invariant relative to cell center
    centroid = coords.mean(axis=0)
    centered_coords = coords - centroid

    # 3. & 4. Distance-based features
    dist_matrix = compute_pbc_distance_matrix(coords, lattice_vectors)

    # Fill diagonal with infinity to ignore self-distance in min/sort operations
    np.fill_diagonal(dist_matrix, np.inf)

    # Chemically-Resolved Reciprocal Proximity
    recip_proximity = np.zeros((num_atoms, Config.NUM_ATOM_TYPES))

    # Identify indices for each atom type
    type_indices = {t: [] for t in Config.ATOM_TYPES}
    for idx, t in enumerate(atom_types):
        if t in type_indices:
            type_indices[t].append(idx)

    for t_idx, t_name in enumerate(Config.ATOM_TYPES):
        indices = type_indices[t_name]
        if not indices:
            # If atom type not present in structure, proximity is 0
            recip_proximity[:, t_idx] = 0.0
        else:
            # Find min distance to any atom of type t_name
            # dist_matrix[:, indices] selects columns corresponding to type t_name
            # min(axis=1) gives min dist for each atom i to any atom of type t_name
            d_min = dist_matrix[:, indices].min(axis=1)

            # Avoid division by zero (though d_min should be > 0 due to diagonal inf)
            # If d_min is inf (e.g. only 1 atom of that type and we masked self), set feat to 0
            with np.errstate(divide="ignore"):
                recip = 1.0 / d_min
            recip[d_min == np.inf] = 0.0
            recip_proximity[:, t_idx] = recip

    # Local Packing Density
    # Mean distance to K nearest neighbors
    # Sort distances for each atom
    sorted_dists = np.sort(dist_matrix, axis=1)
    # Take first K neighbors (columns 0 to K-1, since diagonal is inf and pushed to end)
    # Wait, np.sort pushes np.inf to the end. So sorted_dists[:, 0] is nearest neighbor.
    k = min(Config.K_NEIGHBORS, num_atoms - 1)  # Handle small systems
    if k > 0:
        nearest_k = sorted_dists[:, :k]
        packing_density = nearest_k.mean(axis=1).reshape(-1, 1)
    else:
        packing_density = np.zeros((num_atoms, 1))

    # Concatenate all features
    # Shapes: (N, 4) + (N, 3) + (N, 4) + (N, 1) = (N, 12)
    features = np.hstack([one_hot, centered_coords, recip_proximity, packing_density])

    return features.astype(np.float32)
