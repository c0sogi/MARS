import numpy as np
import os
from library.config import Config


def parse_xyz(file_path):
    """
    Parses the geometry.xyz file to extract lattice vectors and atomic coordinates.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        lattice (np.ndarray): 3x3 array of lattice vectors.
        atom_types (list): List of atomic symbols (str).
        coords (np.ndarray): Nx3 array of Cartesian coordinates.
    """
    lattice_vectors = []
    atom_types = []
    atom_coords = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "lattice_vector":
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                # Format: atom x y z type
                atom_coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    # Fix: Ensure coords has shape (N, 3) even if empty
    # Cite debug_lesson_4
    coords = np.array(atom_coords)
    if coords.size == 0:
        coords = coords.reshape(0, 3)

    return np.array(lattice_vectors), atom_types, coords


def get_pbc_neighbors(coords, lattice, k):
    """
    Finds the K nearest neighbors for each atom in a periodic system using a
    3x3x3 supercell expansion.

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.
        lattice (np.ndarray): 3x3 array of lattice vectors.
        k (int): Number of neighbors to find.

    Returns:
        neighbor_dists (np.ndarray): NxK array of distances to the K nearest neighbors.
        neighbor_indices (np.ndarray): NxK array of indices of the neighbors in the
                                       original atom list (0 to N-1).
    """
    n_atoms = len(coords)

    # Generate translation vectors for 3x3x3 supercell (indices -1, 0, 1)
    ranges = [-1, 0, 1]
    translations = []
    for x in ranges:
        for y in ranges:
            for z in ranges:
                trans = x * lattice[0] + y * lattice[1] + z * lattice[2]
                translations.append(trans)
    translations = np.array(translations)  # Shape: (27, 3)

    # Create supercell coordinates
    # We replicate the atoms 27 times.
    # supercell_coords: (27*N, 3)
    # supercell_indices: (27*N,) - maps back to original atom index 0..N-1
    supercell_coords = []
    supercell_indices = []

    for t in translations:
        shifted_coords = coords + t
        supercell_coords.append(shifted_coords)
        supercell_indices.append(np.arange(n_atoms))

    # Handle empty coords case
    if len(supercell_coords) > 0:
        supercell_coords = np.vstack(supercell_coords)
        supercell_indices = np.concatenate(supercell_indices)
    else:
        supercell_coords = np.empty((0, 3))
        supercell_indices = np.empty((0,), dtype=int)

    # Compute distances and find neighbors
    all_neighbor_dists = []
    all_neighbor_indices = []

    for i in range(n_atoms):
        # Vectorized distance calculation from atom i to all supercell atoms
        diff = supercell_coords - coords[i]
        dists = np.sqrt(np.sum(diff**2, axis=1))

        # Filter out self-interaction (distance ~ 0)
        # We use a small epsilon. Note: An atom is its own neighbor at dist 0 in the (0,0,0) image.
        mask = dists > 1e-6
        valid_dists = dists[mask]
        valid_indices = supercell_indices[mask]

        # Select top K nearest neighbors
        # Use argpartition for efficiency, then sort the top K
        if len(valid_dists) >= k:
            idx_partition = np.argpartition(valid_dists, k)[:k]
            # Sort to ensure d_min is at index 0
            idx_sorted = idx_partition[np.argsort(valid_dists[idx_partition])]

            k_dists = valid_dists[idx_sorted]
            k_indices = valid_indices[idx_sorted]
        else:
            # Fallback for very small systems (unlikely given K=12 and 27 images)
            # Just sort all available
            idx_sorted = np.argsort(valid_dists)
            k_dists = valid_dists[idx_sorted]
            k_indices = valid_indices[idx_sorted]

            # Pad if necessary (though not expected for K=12)
            if len(k_dists) < k:
                pad_len = k - len(k_dists)
                k_dists = np.pad(
                    k_dists, (0, pad_len), "constant", constant_values=1000.0
                )
                k_indices = np.pad(k_indices, (0, pad_len), "edge")

        all_neighbor_dists.append(k_dists)
        all_neighbor_indices.append(k_indices)

    # Handle case where n_atoms is 0
    if n_atoms == 0:
        return np.empty((0, k)), np.empty((0, k))

    return np.array(all_neighbor_dists), np.array(all_neighbor_indices)


def compute_atomic_features(atom_types, coords, lattice, k_neighbors=12):
    """
    Computes the dense, physics-informed feature vector for each atom.

    Features (13 dims):
    1. Identity (One-Hot): 4 dims [Al, Ga, In, O]
    2. Centered Coords (x,y,z): 3 dims
    3. Min Neighbor Distance (d_min): 1 dim
    4. Mean Neighbor Distance (d_mean): 1 dim
    5. Soft Chemical Context (weighted composition): 4 dims

    Args:
        atom_types (list): List of atomic symbols.
        coords (np.ndarray): Nx3 array of Cartesian coordinates.
        lattice (np.ndarray): 3x3 array of lattice vectors.
        k_neighbors (int): Number of neighbors for context calculation.

    Returns:
        features (np.ndarray): Nx13 array of atomic features.
    """
    n_atoms = len(atom_types)
    atom_map = Config.ATOM_MAP

    # 1. Identity One-Hot Encoding
    identity_feats = np.zeros((n_atoms, 4), dtype=np.float32)
    for i, at in enumerate(atom_types):
        if at in atom_map:
            identity_feats[i, atom_map[at]] = 1.0

    # 2. Centered Coordinates
    # Subtract centroid to make coordinates relative to the cluster center
    if len(coords) > 0:
        centroid = np.mean(coords, axis=0)
        centered_coords = coords - centroid
    else:
        centered_coords = coords  # (0, 3)

    # 3, 4, 5. Neighbor-based Features
    # Get distances and indices of K nearest neighbors (PBC corrected)
    neighbor_dists, neighbor_indices = get_pbc_neighbors(coords, lattice, k_neighbors)

    if n_atoms > 0:
        # d_min: Distance to the single closest neighbor
        d_min = neighbor_dists[:, 0].reshape(-1, 1)

        # d_mean: Average distance to the K closest neighbors (local packing density proxy)
        d_mean = np.mean(neighbor_dists, axis=1).reshape(-1, 1)

        # Soft Chemical Context
        # Calculate inverse-distance weights
        # Add epsilon to avoid division by zero (though dists > 1e-6)
        weights = 1.0 / (neighbor_dists + 1e-6)

        # Normalize weights for each atom so they sum to 1
        sum_weights = np.sum(weights, axis=1, keepdims=True)
        norm_weights = weights / sum_weights  # Shape: (N, K)

        # Compute weighted sum of neighbor identities
        context_feats = np.zeros((n_atoms, 4), dtype=np.float32)

        for i in range(n_atoms):
            # Indices of neighbors for atom i
            idxs = neighbor_indices[i]

            # Get one-hot vectors of these neighbors
            # identity_feats is (N, 4), idxs is (K,) -> neighbor_identities is (K, 4)
            neighbor_identities = identity_feats[idxs]

            # Get weights for this atom's neighbors
            w = norm_weights[i].reshape(-1, 1)  # (K, 1)

            # Weighted sum: sum(w * identity) -> (4,)
            weighted_sum = np.sum(neighbor_identities * w, axis=0)
            context_feats[i] = weighted_sum
    else:
        d_min = np.empty((0, 1))
        d_mean = np.empty((0, 1))
        context_feats = np.empty((0, 4))

    # Concatenate all features
    # [Identity(4), Coords(3), d_min(1), d_mean(1), Context(4)] -> 13 dims
    features = np.hstack(
        [identity_feats, centered_coords, d_min, d_mean, context_feats]
    )

    return features.astype(np.float32)
