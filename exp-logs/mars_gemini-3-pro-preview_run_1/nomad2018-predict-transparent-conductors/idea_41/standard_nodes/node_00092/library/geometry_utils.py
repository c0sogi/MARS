import numpy as np
import os
import itertools
from library.config import Config


def parse_xyz(file_path):
    """
    Parses a geometry.xyz file to extract lattice vectors, atomic species, and coordinates.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        lattice (np.ndarray): 3x3 array of lattice vectors.
        species (list): List of atomic species symbols (str).
        coords (np.ndarray): Nx3 array of atomic Cartesian coordinates.
    """
    lattice = []
    species = []
    coords = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "lattice_vector":
            lattice.append([float(x) for x in parts[1:4]])
        elif parts[0] == "atom":
            coords.append([float(x) for x in parts[1:4]])
            species.append(parts[4])

    coords_arr = np.array(coords)
    # Ensure shape is (N, 3) even if N=0. Cite debug_lesson_4
    if coords_arr.ndim == 1:
        coords_arr = coords_arr.reshape(-1, 3)

    return np.array(lattice), species, coords_arr


def center_coordinates(coords):
    """
    Centers atomic coordinates by subtracting the centroid of the unit cell atoms.
    This normalizes the spatial features relative to the structure's center of mass.

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.

    Returns:
        np.ndarray: Centered Nx3 coordinates.
    """
    if coords.shape[0] == 0:
        return coords

    centroid = np.mean(coords, axis=0)
    return coords - centroid


def get_pbc_distances(coords, lattice):
    """
    Computes the pairwise Euclidean distance matrix respecting Periodic Boundary Conditions (PBC).
    Uses a robust 27-image search (3x3x3 supercell) to find the minimum image distance,
    which works correctly for arbitrary (including highly skewed) triclinic unit cells.

    Args:
        coords (np.ndarray): Nx3 atomic coordinates.
        lattice (np.ndarray): 3x3 lattice vectors.

    Returns:
        np.ndarray: NxN distance matrix.
    """
    N = coords.shape[0]

    if N == 0:
        return np.zeros((0, 0), dtype=coords.dtype)

    # Compute pairwise difference vectors between all atoms in the primary cell
    # Shape: (N, N, 3)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    # Generate all possible image shifts for the 27 nearest cells (indices -1, 0, 1)
    # Shape: (27, 3)
    shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3)))

    # Convert fractional shifts to Cartesian shifts using lattice vectors
    # lattice rows are vectors v1, v2, v3
    shift_cart = shifts @ lattice

    # Broadcast add shifts to the pairwise differences
    # diff: (N, N, 1, 3)
    # shift_cart: (1, 1, 27, 3)
    # Result diff_images: (N, N, 27, 3)
    diff_images = diff[:, :, np.newaxis, :] + shift_cart[np.newaxis, np.newaxis, :, :]

    # Compute squared Euclidean distances for all images
    # Shape: (N, N, 27)
    dist_sq = np.sum(diff_images**2, axis=-1)

    # Find the minimum squared distance across all 27 images for each pair
    # Shape: (N, N)
    min_dist_sq = np.min(dist_sq, axis=2)

    # Return actual distances
    return np.sqrt(min_dist_sq)


def compute_local_context(distances, species):
    """
    Computes multi-scale local chemical context features for each atom.

    Generates:
    1. Short-Range Context: Inverse-distance weighted composition of K_SHORT nearest neighbors.
    2. Medium-Range Context: Inverse-distance weighted composition of K_MED nearest neighbors.
    3. Nearest Neighbor Distance: Distance to the single closest atom.

    Args:
        distances (np.ndarray): NxN PBC distance matrix.
        species (list): List of N atomic species symbols.

    Returns:
        short_context (np.ndarray): Nx4 normalized feature vectors.
        med_context (np.ndarray): Nx4 normalized feature vectors.
        nn_dist (np.ndarray): Nx1 vector of nearest neighbor distances.
    """
    N = len(species)

    # Map species symbols to integer indices based on Config
    spec_map = {s: i for i, s in enumerate(Config.ATOMIC_SPECIES)}
    spec_indices = np.array([spec_map[s] for s in species])

    # Initialize feature arrays
    short_context = np.zeros((N, Config.NUM_SPECIES), dtype=np.float32)
    med_context = np.zeros((N, Config.NUM_SPECIES), dtype=np.float32)
    nn_dist = np.zeros((N, 1), dtype=np.float32)

    if N == 0:
        return short_context, med_context, nn_dist

    # Work on a copy to avoid modifying the input matrix
    dists_copy = distances.copy()

    # Mask self-distances (diagonal) with infinity so they aren't picked as neighbors
    np.fill_diagonal(dists_copy, np.inf)

    for i in range(N):
        # Get distances from atom i to all others
        d_i = dists_copy[i]

        # Sort indices by distance (ascending)
        sorted_idx = np.argsort(d_i)

        # 1. Nearest Neighbor Distance
        # The first element in sorted_idx is the closest neighbor
        nn_dist[i] = d_i[sorted_idx[0]]

        # 2. Short-Range Context
        # Select top K_SHORT neighbors
        k_s = min(Config.K_SHORT, N - 1)
        if k_s > 0:
            nbr_idx_s = sorted_idx[:k_s]
            nbr_dists_s = d_i[nbr_idx_s]
            nbr_specs_s = spec_indices[nbr_idx_s]

            # Calculate weights: 1 / (d + epsilon)
            weights_s = 1.0 / (nbr_dists_s + 1e-6)

            # Aggregate weighted one-hot vectors
            for j, s_idx in enumerate(nbr_specs_s):
                short_context[i, s_idx] += weights_s[j]

            # Normalize to sum to 1 (encodes identity/composition, not density)
            total_w_s = np.sum(weights_s)
            if total_w_s > 1e-9:
                short_context[i] /= total_w_s

        # 3. Medium-Range Context
        # Select top K_MED neighbors
        k_m = min(Config.K_MED, N - 1)
        if k_m > 0:
            nbr_idx_m = sorted_idx[:k_m]
            nbr_dists_m = d_i[nbr_idx_m]
            nbr_specs_m = spec_indices[nbr_idx_m]

            weights_m = 1.0 / (nbr_dists_m + 1e-6)

            for j, s_idx in enumerate(nbr_specs_m):
                med_context[i, s_idx] += weights_m[j]

            total_w_m = np.sum(weights_m)
            if total_w_m > 1e-9:
                med_context[i] /= total_w_m

    return short_context, med_context, nn_dist
