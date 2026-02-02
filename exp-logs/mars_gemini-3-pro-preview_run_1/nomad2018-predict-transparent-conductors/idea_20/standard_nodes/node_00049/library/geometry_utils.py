import numpy as np
import os
from library.config import Config


def parse_xyz(file_path):
    """
    Parses the custom XYZ format to extract lattice vectors and atomic information.

    Args:
        file_path (str): Path to the geometry.xyz file.

    Returns:
        tuple:
            - lattice_vectors (np.ndarray): 3x3 matrix of lattice vectors.
            - atom_types (list): List of atomic symbols.
            - coords (np.ndarray): Nx3 matrix of atomic Cartesian coordinates.
    """
    lattice_vectors = []
    atom_types = []
    coords = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if parts[0] == "lattice_vector":
                lattice_vectors.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    return np.array(lattice_vectors), atom_types, np.array(coords)


def get_cell_volume(lattice_vectors):
    """
    Computes the volume of the unit cell using the scalar triple product.

    Args:
        lattice_vectors (np.ndarray): 3x3 matrix of lattice vectors.

    Returns:
        float: Volume of the unit cell.
    """
    return np.abs(
        np.dot(lattice_vectors[0], np.cross(lattice_vectors[1], lattice_vectors[2]))
    )


def get_atomic_density(num_atoms, volume):
    """
    Computes the atomic density.

    Args:
        num_atoms (int): Total number of atoms.
        volume (float): Volume of the unit cell.

    Returns:
        float: Atomic density.
    """
    return num_atoms / volume


def get_pbc_displacement(r_i, r_j, lattice_vectors):
    """
    Calculates the displacement vector between two points under Periodic Boundary Conditions (MIC).

    Args:
        r_i (np.ndarray): Coordinates of point i.
        r_j (np.ndarray): Coordinates of point j.
        lattice_vectors (np.ndarray): 3x3 matrix of lattice vectors.

    Returns:
        np.ndarray: Displacement vector (r_j - r_i) respecting PBC.
    """
    diff = r_j - r_i
    # Convert to fractional coordinates
    # lattice_vectors rows are a1, a2, a3.
    # r = n1*a1 + n2*a2 + n3*a3
    # r * inv(L) = n
    inv_lattice = np.linalg.inv(lattice_vectors.T)  # Transpose because vectors are rows
    frac_diff = np.dot(inv_lattice, diff)

    # Apply MIC in fractional space
    frac_diff -= np.round(frac_diff)

    # Convert back to Cartesian
    cart_diff = np.dot(lattice_vectors.T, frac_diff)
    return cart_diff


def compute_local_anisotropy(coords, lattice_vectors, k_neighbors=12):
    """
    Computes local anisotropy eigenvalues and nearest neighbor distances for each atom.

    Args:
        coords (np.ndarray): Nx3 matrix of atomic coordinates.
        lattice_vectors (np.ndarray): 3x3 matrix of lattice vectors.
        k_neighbors (int): Number of nearest neighbors to consider.

    Returns:
        tuple:
            - eigenvalues (np.ndarray): Nx3 matrix of covariance eigenvalues (sorted).
            - nn_dists (np.ndarray): Nx1 array of distances to the nearest neighbor.
    """
    n_atoms = len(coords)

    # Generate supercell shifts (3x3x3)
    # This ensures we find the true K nearest neighbors even in small cells or skewed cells
    shifts = []
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            for z in [-1, 0, 1]:
                shifts.append(
                    x * lattice_vectors[0]
                    + y * lattice_vectors[1]
                    + z * lattice_vectors[2]
                )
    shifts = np.array(shifts)  # 27 x 3

    # Replicate atoms in the supercell
    # super_coords shape: (27 * N, 3)
    # We broadcast: coords (N, 1, 3) + shifts (1, 27, 3) -> (N, 27, 3) -> reshape
    super_coords = (coords[:, np.newaxis, :] + shifts[np.newaxis, :, :]).reshape(-1, 3)

    eigenvalues_list = []
    nn_dists_list = []

    for i in range(n_atoms):
        center_atom = coords[i]

        # Compute distances to all atoms in the supercell
        diffs = super_coords - center_atom
        dists_sq = np.sum(diffs**2, axis=1)

        # Filter out self (distance ~ 0)
        mask = dists_sq > 1e-6
        valid_dists_sq = dists_sq[mask]
        valid_diffs = diffs[mask]

        # Sort to find K nearest
        if len(valid_dists_sq) < k_neighbors:
            k_eff = len(valid_dists_sq)
        else:
            k_eff = k_neighbors

        # partition is faster than sort for finding top K
        if k_eff > 0:
            nearest_indices = np.argpartition(valid_dists_sq, k_eff - 1)[:k_eff]

            # Get the vectors to these neighbors
            neighbor_vectors = valid_diffs[nearest_indices]  # (K, 3)

            # Nearest neighbor distance (min of the K)
            min_dist = np.sqrt(np.min(valid_dists_sq[nearest_indices]))

            # Compute Covariance Matrix
            # C = (1/K) * sum(v * v.T)
            covariance = np.dot(neighbor_vectors.T, neighbor_vectors) / k_eff

            # Compute Eigenvalues
            evals = np.linalg.eigvalsh(covariance)
            # Sort eigenvalues ascending
            evals.sort()
        else:
            min_dist = 0.0
            evals = np.zeros(3)

        nn_dists_list.append(min_dist)
        eigenvalues_list.append(evals)

    return np.array(eigenvalues_list), np.array(nn_dists_list).reshape(-1, 1)


def get_centered_coordinates(coords, lattice_vectors):
    """
    Centers the atomic coordinates relative to the unit cell centroid.

    Args:
        coords (np.ndarray): Nx3 matrix of atomic coordinates.
        lattice_vectors (np.ndarray): 3x3 matrix of lattice vectors.

    Returns:
        np.ndarray: Centered coordinates.
    """
    # Unit cell centroid is the center of the parallelepiped defined by lattice vectors
    # Origin is (0,0,0), so center is 0.5 * (a + b + c)
    centroid = 0.5 * np.sum(lattice_vectors, axis=0)
    return coords - centroid
