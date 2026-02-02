import numpy as np
import torch
import os


def parse_xyz(file_path):
    """
    Parses a geometry.xyz file to extract lattice vectors, atomic positions, and types.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        lattice_matrix (np.ndarray): 3x3 array of lattice vectors.
        atom_types (list): List of atomic symbols.
        coords (np.ndarray): Nx3 array of Cartesian atomic coordinates.
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
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                # format: atom x y z symbol
                coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    return np.array(lattice_vectors), atom_types, np.array(coords)


def get_lattice_params(lattice_matrix):
    """
    Computes lattice lengths (a, b, c) and angles (alpha, beta, gamma) from the lattice matrix.

    Args:
        lattice_matrix (np.ndarray): 3x3 matrix where rows are lattice vectors v1, v2, v3.

    Returns:
        lengths (np.ndarray): [a, b, c]
        angles (np.ndarray): [alpha, beta, gamma] in degrees.
    """
    v1 = lattice_matrix[0]
    v2 = lattice_matrix[1]
    v3 = lattice_matrix[2]

    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    c = np.linalg.norm(v3)

    alpha_rad = np.arccos(np.clip(np.dot(v2, v3) / (b * c), -1.0, 1.0))
    beta_rad = np.arccos(np.clip(np.dot(v1, v3) / (a * c), -1.0, 1.0))
    gamma_rad = np.arccos(np.clip(np.dot(v1, v2) / (a * b), -1.0, 1.0))

    angles = np.degrees([alpha_rad, beta_rad, gamma_rad])
    return np.array([a, b, c]), angles


def get_cell_volume(lattice_matrix):
    """
    Computes the volume of the unit cell.

    Args:
        lattice_matrix (np.ndarray): 3x3 lattice matrix.

    Returns:
        float: Volume of the unit cell.
    """
    return np.abs(np.linalg.det(lattice_matrix))


def cartesian_to_fractional(coords, lattice_matrix):
    """
    Converts Cartesian coordinates to fractional coordinates.

    Args:
        coords (np.ndarray): Nx3 Cartesian coordinates.
        lattice_matrix (np.ndarray): 3x3 lattice matrix.

    Returns:
        np.ndarray: Nx3 fractional coordinates.
    """
    # r = u*v1 + v*v2 + w*v3
    # r = [u, v, w] @ Lattice_Matrix
    # [u, v, w] = r @ inv(Lattice_Matrix)
    inv_lattice = np.linalg.inv(lattice_matrix)
    return coords @ inv_lattice


def compute_pbc_distances(coords, lattice_matrix):
    """
    Computes pairwise distances respecting Periodic Boundary Conditions (PBC).
    Uses the Minimum Image Convention on fractional coordinates.

    Args:
        coords (np.ndarray): Nx3 Cartesian coordinates.
        lattice_matrix (np.ndarray): 3x3 lattice matrix.

    Returns:
        np.ndarray: NxN distance matrix.
    """
    n_atoms = len(coords)
    frac_coords = cartesian_to_fractional(coords, lattice_matrix)

    # Compute difference in fractional coordinates: diff[i, j, :] = frac[i] - frac[j]
    # Shape: (N, N, 3)
    diff_frac = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]

    # Apply Minimum Image Convention: wrap differences to [-0.5, 0.5]
    diff_frac -= np.round(diff_frac)

    # Convert back to Cartesian to get actual distances
    # Shape: (N, N, 3)
    diff_cart = diff_frac @ lattice_matrix

    # Compute Euclidean norm along the last axis
    dist_matrix = np.linalg.norm(diff_cart, axis=2)

    return dist_matrix


def compute_local_potential(dist_matrix):
    """
    Computes the scalar potential proxy for each atom.
    P_i = sum_{j != i} (1 / d_ij)

    Args:
        dist_matrix (np.ndarray): NxN distance matrix.

    Returns:
        np.ndarray: Array of shape (N,) containing potential values.
    """
    # Avoid division by zero on diagonal
    # Create a masked matrix or add infinity to diagonal
    n = dist_matrix.shape[0]
    if n <= 1:
        return np.zeros(n)

    # Copy to avoid modifying input
    D = dist_matrix.copy()

    # Set diagonal to infinity so 1/D becomes 0
    np.fill_diagonal(D, np.inf)

    # Compute sum of inverse distances
    # Handle potential zeros off-diagonal if any (shouldn't happen for distinct atoms)
    with np.errstate(divide="ignore"):
        inv_dist = 1.0 / D

    potential = np.sum(inv_dist, axis=1)
    return potential


def center_coordinates(coords):
    """
    Centers the coordinates by subtracting the centroid.

    Args:
        coords (np.ndarray): Nx3 Cartesian coordinates.

    Returns:
        np.ndarray: Centered coordinates.
    """
    centroid = np.mean(coords, axis=0)
    return coords - centroid
