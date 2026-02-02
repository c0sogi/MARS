import numpy as np
import os


def parse_xyz(file_path):
    """
    Parses a geometry.xyz file to extract lattice vectors and atomic information.

    Args:
        file_path (str): Path to the .xyz file.

    Returns:
        tuple:
            - lattice_matrix (np.ndarray): 3x3 matrix of lattice vectors.
            - atom_types (list): List of atomic symbols (str).
            - cart_coords (np.ndarray): Nx3 matrix of Cartesian coordinates.
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
                # format: atom x y z symbol
                atom_coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    # Fix for empty arrays (Cite debug_lesson_4)
    cart_coords = np.array(atom_coords)
    if cart_coords.size == 0:
        cart_coords = cart_coords.reshape(0, 3)

    return np.array(lattice_vectors), atom_types, cart_coords


def compute_fractional_coords(cart_coords, lattice_matrix):
    """
    Transforms Cartesian coordinates to fractional coordinates using the lattice matrix.

    Args:
        cart_coords (np.ndarray): Nx3 matrix of Cartesian coordinates.
        lattice_matrix (np.ndarray): 3x3 matrix where rows are lattice vectors.

    Returns:
        np.ndarray: Nx3 matrix of fractional coordinates.
    """
    # Solves X_frac * Lattice = X_cart
    inv_lattice = np.linalg.inv(lattice_matrix)
    frac_coords = cart_coords @ inv_lattice
    return frac_coords


def compute_pbc_distances(frac_coords, lattice_matrix):
    """
    Computes pairwise distances respecting periodic boundary conditions using the
    minimum image convention.

    Args:
        frac_coords (np.ndarray): Nx3 matrix of fractional coordinates.
        lattice_matrix (np.ndarray): 3x3 matrix of lattice vectors.

    Returns:
        np.ndarray: NxN matrix of pairwise distances.
    """
    # Compute pairwise differences in fractional coordinates
    # diff[i, j, :] = frac_coords[i] - frac_coords[j]
    # Broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff_frac = frac_coords[:, np.newaxis, :] - frac_coords[np.newaxis, :, :]

    # Apply Minimum Image Convention: wrap fractional differences to [-0.5, 0.5]
    diff_frac -= np.round(diff_frac)

    # Convert back to Cartesian space
    # diff_cart[i, j] = diff_frac[i, j] @ lattice_matrix
    diff_cart = diff_frac @ lattice_matrix

    # Compute Euclidean distances
    distances = np.sqrt(np.sum(diff_cart**2, axis=-1))

    return distances


def extract_atomic_features(file_path):
    """
    Extracts atomic features from an XYZ file, including centered coordinates,
    fractional coordinates, nearest neighbor distances, and local potential proxies.

    Args:
        file_path (str): Path to the geometry file.

    Returns:
        dict: Dictionary containing:
            - 'atom_types': List of atom symbols.
            - 'centered_cart_coords': Nx3 np.ndarray.
            - 'frac_coords': Nx3 np.ndarray.
            - 'nn_dist': Nx1 np.ndarray (nearest neighbor distance).
            - 'potential_proxy': Nx1 np.ndarray (sum of inverse distances).
            - 'lattice_matrix': 3x3 np.ndarray.
            - 'num_atoms': int.
    """
    # 1. Parse Data
    lattice_matrix, atom_types, cart_coords = parse_xyz(file_path)
    num_atoms = len(atom_types)

    # 2. Centered Cartesian Coordinates
    # Cite debug_lesson_6: Position Guard Clauses Before Unsafe Operations
    if num_atoms > 0:
        centroid = np.mean(cart_coords, axis=0)
        centered_cart_coords = cart_coords - centroid
    else:
        centered_cart_coords = cart_coords

    # 3. Fractional Coordinates
    frac_coords = compute_fractional_coords(cart_coords, lattice_matrix)

    # 4. PBC Distances
    dist_matrix = compute_pbc_distances(frac_coords, lattice_matrix)

    # Handle self-distance (diagonal is 0) for feature extraction
    # Replace diagonal with infinity to ignore it in min() and 1/x calculations
    np.fill_diagonal(dist_matrix, np.inf)

    # 5. Nearest Neighbor Distance
    if num_atoms > 0:
        nn_dist = np.min(dist_matrix, axis=1).reshape(-1, 1)

        # 6. Local Potential Proxy (Sum of 1/r)
        # 1/inf = 0, so self-interaction contributes 0
        inv_dist = 1.0 / dist_matrix
        potential_proxy = np.sum(inv_dist, axis=1).reshape(-1, 1)
    else:
        nn_dist = np.zeros((0, 1))
        potential_proxy = np.zeros((0, 1))

    return {
        "atom_types": atom_types,
        "centered_cart_coords": centered_cart_coords,
        "frac_coords": frac_coords,
        "nn_dist": nn_dist,
        "potential_proxy": potential_proxy,
        "lattice_matrix": lattice_matrix,
        "num_atoms": num_atoms,
    }
