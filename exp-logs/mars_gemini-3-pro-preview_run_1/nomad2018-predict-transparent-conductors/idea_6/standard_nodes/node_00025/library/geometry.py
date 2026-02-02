import numpy as np
import os


def parse_xyz(file_path):
    """
    Parses an XYZ file to extract atom types, coordinates, and lattice vectors.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        tuple: (atoms, coords, lattice_vectors)
            - atoms (np.ndarray): Array of atom symbols (str).
            - coords (np.ndarray): Array of atomic coordinates (float) of shape (N, 3).
            - lattice_vectors (np.ndarray): Array of lattice vectors (float) of shape (3, 3).
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    atoms = []
    coords = []
    lattice_vectors = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "atom":
            # Format: atom x y z symbol
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            atoms.append(parts[4])
        elif parts[0] == "lattice_vector":
            # Format: lattice_vector x y z
            lattice_vectors.append([float(parts[1]), float(parts[2]), float(parts[3])])

    # Handle empty case to ensure consistent shapes
    if len(atoms) == 0:
        return (
            np.array([], dtype=str),
            np.empty((0, 3), dtype=float),
            (
                np.array(lattice_vectors)
                if len(lattice_vectors) > 0
                else np.empty((0, 3), dtype=float)
            ),
        )

    return np.array(atoms), np.array(coords), np.array(lattice_vectors)


def get_pbc_distances(coords, lattice_vectors):
    """
    Computes pairwise distances under periodic boundary conditions using the
    minimum image convention.

    Args:
        coords (np.ndarray): Atomic coordinates of shape (N, 3).
        lattice_vectors (np.ndarray): Lattice vectors of shape (3, 3).

    Returns:
        np.ndarray: Pairwise distance matrix of shape (N, N).
    """
    n_atoms = len(coords)
    if n_atoms == 0:
        return np.zeros((0, 0))
    if n_atoms == 1:
        return np.zeros((1, 1))

    # Generate 27 images (center + 26 neighbors)
    images = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                images.append(
                    i * lattice_vectors[0]
                    + j * lattice_vectors[1]
                    + k * lattice_vectors[2]
                )
    images = np.array(images)  # Shape: (27, 3)

    # Compute distances: d_ij = min(|r_i - r_j + T|)
    # Expansion: (N, 1, 1, 3) - (1, N, 1, 3) + (1, 1, 27, 3) -> (N, N, 27, 3)
    diff = (
        coords[:, None, None, :] - coords[None, :, None, :] + images[None, None, :, :]
    )

    # Squared distances
    dists_sq = np.sum(diff**2, axis=-1)  # Shape: (N, N, 27)

    # Minimum squared distance over all images
    min_dists_sq = np.min(dists_sq, axis=-1)  # Shape: (N, N)

    return np.sqrt(min_dists_sq)


def calculate_cell_volume(lattice_vectors):
    """
    Calculates the volume of the unit cell defined by lattice vectors.

    Args:
        lattice_vectors (np.ndarray): Lattice vectors of shape (3, 3).

    Returns:
        float: Volume of the unit cell.
    """
    # Volume is the scalar triple product: |v1 . (v2 x v3)|
    return np.abs(
        np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
    )


def compute_local_fingerprint(coords, lattice_vectors, k=12):
    """
    Computes statistical summary of local neighborhood for each atom.
    Simplified to just nearest neighbor distance (Cite solution_lesson_node_00021).

    Args:
        coords (np.ndarray): Atomic coordinates of shape (N, 3).
        lattice_vectors (np.ndarray): Lattice vectors of shape (3, 3).
        k (int): Number of nearest neighbors to consider (unused in simplified version but kept for signature).

    Returns:
        np.ndarray: Array of shape (N, 1) containing [d_min] for each atom.
    """
    n_atoms = len(coords)
    if n_atoms == 0:
        return np.zeros((0, 1))

    dist_matrix = get_pbc_distances(coords, lattice_vectors)
    fingerprints = []

    # Fill diagonal with infinity to ignore self-distance in sorting
    np.fill_diagonal(dist_matrix, np.inf)

    for i in range(n_atoms):
        # Min distance (nearest neighbor)
        # If n_atoms > 1, min is the nearest neighbor.
        # If n_atoms == 1, min of empty/inf is inf, handle gracefully.
        if n_atoms > 1:
            d_min = np.min(dist_matrix[i])
        else:
            d_min = 0.0

        fingerprints.append([d_min])

    return np.array(fingerprints)
