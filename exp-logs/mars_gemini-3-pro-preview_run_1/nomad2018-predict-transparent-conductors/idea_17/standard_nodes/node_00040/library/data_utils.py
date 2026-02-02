import os
import numpy as np
import pandas as pd
from library.config import ATOM_TO_IDX


def parse_xyz(file_path):
    """
    Parses an XYZ file to extract lattice vectors, atomic species, and coordinates.

    Args:
        file_path (str): Path to the geometry.xyz file.

    Returns:
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.
        atom_species (list): List of atomic species strings.
        atom_coords (np.ndarray): Nx3 array of atomic coordinates.
    """
    lattice_vectors = []
    atom_species = []
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
            # format: atom x y z species
            atom_coords.append([float(x) for x in parts[1:4]])
            atom_species.append(parts[4])

    return np.array(lattice_vectors), atom_species, np.array(atom_coords)


def calculate_cell_volume(lattice_vectors):
    """
    Calculates the volume of the unit cell.

    Args:
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.

    Returns:
        float: Volume of the unit cell.
    """
    return np.abs(np.linalg.det(lattice_vectors))


def compute_pbc_distance_matrix(coords, lattice_vectors):
    """
    Computes pairwise distances respecting Periodic Boundary Conditions (Minimum Image Convention).

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.
        lattice_vectors (np.ndarray): 3x3 array of lattice vectors.

    Returns:
        np.ndarray: NxN distance matrix.
    """
    n_atoms = len(coords)

    # Compute difference vectors between all pairs: (N, N, 3)
    # diff[i, j] = coords[i] - coords[j]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    # Convert to fractional coordinates
    # r_cart = frac @ lattice
    # frac = r_cart @ lattice_inv
    try:
        lattice_inv = np.linalg.inv(lattice_vectors)
    except np.linalg.LinAlgError:
        # Fallback for singular matrix (should not happen in valid data)
        return np.linalg.norm(diff, axis=-1)

    # Transform differences to fractional space
    diff_frac = diff @ lattice_inv

    # Apply Minimum Image Convention in fractional space
    # Round to nearest integer to find the nearest image shift
    diff_frac_mic = diff_frac - np.round(diff_frac)

    # Convert back to Cartesian coordinates
    diff_cart_mic = diff_frac_mic @ lattice_vectors

    # Compute Euclidean distances
    dist_matrix = np.linalg.norm(diff_cart_mic, axis=-1)

    return dist_matrix


def get_atomic_features(file_path):
    """
    Extracts atomic features for the PC-WDS model.

    Features per atom (9 dims):
    1-4: One-hot encoding (Al, Ga, In, O)
    5-7: Centered Cartesian coordinates (x, y, z)
    8: Nearest Neighbor Distance
    9: Local Potential Proxy (sum of 1/d)

    Args:
        file_path (str): Path to the geometry file.

    Returns:
        np.ndarray: Nx9 feature matrix.
    """
    lattice_vectors, species, coords = parse_xyz(file_path)
    n_atoms = len(species)

    # 1. Atomic Identity (One-hot)
    one_hot = np.zeros((n_atoms, 4), dtype=np.float32)
    for i, s in enumerate(species):
        if s in ATOM_TO_IDX:
            one_hot[i, ATOM_TO_IDX[s]] = 1.0

    # 2. Centered Coordinates
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid

    # 3. PBC Distance Matrix
    dist_matrix = compute_pbc_distance_matrix(coords, lattice_vectors)

    # Mask diagonal for calculations (distance to self is 0)
    np.fill_diagonal(dist_matrix, np.inf)

    # 4. Nearest Neighbor Distance
    nn_dist = np.min(dist_matrix, axis=1).reshape(-1, 1)

    # 5. Local Potential Proxy
    # P_i = sum_{j != i} (1 / d_ij)
    # Avoid division by zero or inf
    with np.errstate(divide="ignore"):
        inv_dist = 1.0 / dist_matrix

    # The diagonal was inf, so 1/inf is 0.0, which is correct for the sum (exclude self)
    potential = np.sum(inv_dist, axis=1).reshape(-1, 1)

    # Concatenate all features
    # Shape: (N, 4 + 3 + 1 + 1) = (N, 9)
    atomic_features = np.hstack([one_hot, centered_coords, nn_dist, potential])

    return atomic_features.astype(np.float32)


def get_global_features(row, lattice_vectors, num_atoms):
    """
    Extracts global features for the PC-WDS model.

    Features (12 dims):
    1-3: Lattice Vector Lengths (from CSV)
    4-6: Lattice Angles (from CSV)
    7: Unit Cell Volume (calculated)
    8: Atomic Density (calculated)
    9-11: Stoichiometry (Al, Ga, In) (from CSV)
    12: Total Number of Atoms (from CSV)

    Args:
        row (pd.Series): Row from the metadata dataframe.
        lattice_vectors (np.ndarray): 3x3 lattice matrix from XYZ.
        num_atoms (int): Total number of atoms.

    Returns:
        np.ndarray: 1D array of global features.
    """
    # Extract from CSV row
    # Lattice lengths
    lv1 = row["lattice_vector_1_ang"]
    lv2 = row["lattice_vector_2_ang"]
    lv3 = row["lattice_vector_3_ang"]

    # Lattice angles
    alpha = row["lattice_angle_alpha_degree"]
    beta = row["lattice_angle_beta_degree"]
    gamma = row["lattice_angle_gamma_degree"]

    # Stoichiometry
    pct_al = row["percent_atom_al"]
    pct_ga = row["percent_atom_ga"]
    pct_in = row["percent_atom_in"]

    # Total atoms (from CSV, should match num_atoms from XYZ)
    total_atoms_csv = row["number_of_total_atoms"]

    # Calculated features
    volume = calculate_cell_volume(lattice_vectors)
    density = num_atoms / volume if volume > 0 else 0.0

    features = np.array(
        [
            lv1,
            lv2,
            lv3,
            alpha,
            beta,
            gamma,
            volume,
            density,
            pct_al,
            pct_ga,
            pct_in,
            total_atoms_csv,
        ],
        dtype=np.float32,
    )

    return features
