import numpy as np
import os


def read_xyz(file_path):
    """
    Parses a .xyz file to extract lattice vectors, atomic types, and Cartesian coordinates.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        lattice_matrix (np.ndarray): 3x3 array of lattice vectors.
        atom_types (list): List of atomic symbols (e.g., ['Ga', 'Al', ...]).
        cartesian_coords (np.ndarray): Nx3 array of atomic positions.
    """
    lattice_vectors = []
    atom_types = []
    cartesian_coords = []

    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == "lattice_vector":
                # Format: lattice_vector x y z
                vec = [float(x) for x in parts[1:4]]
                lattice_vectors.append(vec)
            elif parts[0] == "atom":
                # Format: atom x y z Type
                pos = [float(x) for x in parts[1:4]]
                cartesian_coords.append(pos)
                atom_types.append(parts[4])

    # Cite debug_lesson_3: Enforce Consistent Rank When Creating NumPy Arrays from Potentially Empty Lists
    coords = np.array(cartesian_coords)
    if coords.size == 0:
        coords = coords.reshape(0, 3)
    return np.array(lattice_vectors), atom_types, coords


def lattice_from_params(lengths, angles):
    """
    Constructs a 3x3 lattice matrix from lattice lengths and angles.
    Aligns a along x-axis, b in xy-plane.

    Args:
        lengths (list/tuple): [a, b, c] in Angstroms.
        angles (list/tuple): [alpha, beta, gamma] in degrees.

    Returns:
        np.ndarray: 3x3 lattice matrix.
    """
    a, b, c = lengths
    alpha, beta, gamma = np.radians(angles)

    # Vector a is along the x-axis
    va = np.array([a, 0.0, 0.0])

    # Vector b is in the xy-plane
    vb = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])

    # Vector c has components in x, y, z
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    vc = np.array([cx, cy, cz])

    return np.array([va, vb, vc])


def cartesian_to_fractional(coords, lattice_matrix):
    """
    Converts Cartesian coordinates to fractional coordinates.

    Args:
        coords (np.ndarray): Nx3 Cartesian coordinates.
        lattice_matrix (np.ndarray): 3x3 lattice matrix (row vectors).

    Returns:
        np.ndarray: Nx3 fractional coordinates.
    """
    # F = C * L^-1
    # Cite debug_lesson_3: Enforce Consistent Rank
    if coords.ndim == 1:
        coords = coords.reshape(-1, 3)

    inv_lattice = np.linalg.inv(
        lattice_matrix.T
    )  # Transpose because vectors are rows in lattice_matrix but math usually assumes cols or we do C @ inv(L)
    # Actually, if R = n1*a + n2*b + n3*c (where a,b,c are rows), then R = [n1 n2 n3] @ [a; b; c].
    # So R_cart = R_frac @ Lattice.
    # Therefore R_frac = R_cart @ inv(Lattice).
    return np.dot(coords, np.linalg.inv(lattice_matrix))


def get_pbc_distances(frac_coords, lattice_matrix):
    """
    Computes pairwise distances between atoms respecting Periodic Boundary Conditions
    using the Minimum Image Convention.

    Args:
        frac_coords (np.ndarray): Nx3 fractional coordinates.
        lattice_matrix (np.ndarray): 3x3 lattice matrix.

    Returns:
        np.ndarray: NxN distance matrix.
    """
    n_atoms = len(frac_coords)

    # Calculate difference vectors in fractional coordinates: diff[i, j] = frac[i] - frac[j]
    # shape: (N, N, 3)
    diff_frac = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]

    # Apply Minimum Image Convention: round to nearest integer to find nearest image
    # d_mic = d - round(d)
    diff_frac -= np.round(diff_frac)

    # Convert back to Cartesian coordinates
    # diff_cart[i, j] = diff_frac[i, j] @ lattice_matrix
    diff_cart = np.dot(diff_frac, lattice_matrix)

    # Compute Euclidean norms
    dist_matrix = np.sqrt(np.sum(diff_cart**2, axis=-1))

    return dist_matrix


def compute_local_potential(dist_matrix):
    """
    Calculates the scalar potential proxy and nearest neighbor distance for each atom.
    Potential P_i = sum_{j != i} (1 / d_ij).

    Args:
        dist_matrix (np.ndarray): NxN pairwise distance matrix.

    Returns:
        potential (np.ndarray): Array of shape (N,) containing potential values.
        nn_dist (np.ndarray): Array of shape (N,) containing nearest neighbor distances.
    """
    # Avoid division by zero on the diagonal
    # Create a masked array or add infinity to diagonal
    n = dist_matrix.shape[0]

    # Cite debug_lesson_11: Explicitly Handle Zero-Cardinality Inputs
    if n == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # Copy to avoid modifying input
    d = dist_matrix.copy()

    # Set diagonal to infinity so it doesn't affect min or inverse sum
    np.fill_diagonal(d, np.inf)

    # Nearest neighbor distance
    nn_dist = np.min(d, axis=1)

    # Potential: sum of 1/d for off-diagonal elements
    # 1/inf is 0, so we can safely invert the whole matrix (handling inf)
    with np.errstate(divide="ignore"):
        inv_dist = 1.0 / d

    potential = np.sum(inv_dist, axis=1)

    return potential, nn_dist
