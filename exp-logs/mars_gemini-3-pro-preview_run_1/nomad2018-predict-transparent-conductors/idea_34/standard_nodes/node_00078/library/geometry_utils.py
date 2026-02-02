import os
import numpy as np
from ase.io import read
from library.config import Config


def parse_xyz(file_path):
    """
    Parses an XYZ file using ASE.

    Args:
        file_path (str): Relative path to the geometry.xyz file from input dir.

    Returns:
        ase.Atoms: The atoms object.
    """
    full_path = os.path.join(Config.INPUT_DIR, file_path)
    # Fallback if full path provided or file exists directly
    if not os.path.exists(full_path) and os.path.exists(file_path):
        full_path = file_path

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found: {full_path}")

    atoms = read(full_path, format="aims")
    return atoms


def get_pbc_neighbors(atoms, k):
    """
    Computes pairwise distances under PBC and identifies K nearest neighbors.

    Args:
        atoms (ase.Atoms): The atoms object.
        k (int): Number of neighbors to find.

    Returns:
        distances (np.ndarray): (N, k) array of distances to k nearest neighbors.
        indices (np.ndarray): (N, k) array of indices of k nearest neighbors.
    """
    # Compute all pairwise distances with Minimum Image Convention
    # This returns an NxN symmetric matrix
    d_matrix = atoms.get_all_distances(mic=True)

    # We want to ignore self-distance (diagonal is 0)
    # Set diagonal to infinity so it sorts to the end
    np.fill_diagonal(d_matrix, np.inf)

    # Sort to find nearest neighbors
    # We need both distances and indices
    sorted_indices = np.argsort(d_matrix, axis=1)
    sorted_distances = np.sort(d_matrix, axis=1)

    # Handle case where N <= k
    n_atoms = len(atoms)
    actual_k = min(k, n_atoms - 1)

    if actual_k <= 0:
        # Case with single atom, though unlikely in this dataset
        return np.zeros((n_atoms, 0)), np.zeros((n_atoms, 0), dtype=int)

    neighbor_indices = sorted_indices[:, :actual_k]
    neighbor_distances = sorted_distances[:, :actual_k]

    return neighbor_distances, neighbor_indices


def compute_local_stoichiometry(neighbor_indices, atomic_symbols, atom_types):
    """
    Calculates the fractional chemical composition of the local neighborhood.

    Args:
        neighbor_indices (np.ndarray): (N, k) indices of neighbors.
        atomic_symbols (list or np.ndarray): Array of symbols for all atoms in the cell.
        atom_types (list): List of possible atom types (e.g. ['Al', 'Ga', 'In', 'O']).

    Returns:
        local_stoich (np.ndarray): (N, len(atom_types)) array of fractions.
    """
    n_atoms = len(atomic_symbols)
    if neighbor_indices.shape[1] == 0:
        return np.zeros((n_atoms, len(atom_types)), dtype=np.float32)

    n_types = len(atom_types)

    # Map symbols to integers 0..3
    type_map = {sym: i for i, sym in enumerate(atom_types)}

    # Convert all symbols in the system to integers
    atom_int_types = np.array([type_map.get(s, -1) for s in atomic_symbols])

    # Get the types of the neighbors
    # neighbor_types shape: (N, k)
    neighbor_types = atom_int_types[neighbor_indices]

    # Count occurrences of each type for each atom
    local_stoich = np.zeros((n_atoms, n_types), dtype=np.float32)

    for i in range(n_types):
        # Sum boolean mask along the neighbor axis (axis 1)
        local_stoich[:, i] = np.sum(neighbor_types == i, axis=1)

    # Normalize by k (number of neighbors) to get fractions
    k = neighbor_indices.shape[1]
    if k > 0:
        local_stoich /= k

    return local_stoich


def compute_global_features(atoms, atom_types):
    """
    Extracts macroscopic properties.

    Args:
        atoms (ase.Atoms): Atoms object.
        atom_types (list): List of atom types for stoichiometry calculation.

    Returns:
        features (np.ndarray): 1D array of global features.
    """
    # 1. Lattice parameters (lengths and angles)
    # cellpar returns [a, b, c, alpha, beta, gamma]
    cell_specific = atoms.cell.cellpar()

    # 2. Volume
    volume = atoms.get_volume()

    # 3. Total Atoms
    n_atoms = len(atoms)

    # 4. Density
    # Avoid division by zero
    density = n_atoms / volume if volume > 1e-6 else 0.0

    # 5. Global Stoichiometry (Al, Ga, In)
    # Config implies 3 stoichiometry features. We assume Al, Ga, In fractions.
    symbols = np.array(atoms.get_chemical_symbols())
    stoich = []

    # We calculate fractions for Al, Ga, In relative to total atoms
    target_elements = ["Al", "Ga", "In"]
    for el in target_elements:
        count = np.sum(symbols == el)
        fraction = count / n_atoms if n_atoms > 0 else 0.0
        stoich.append(fraction)

    # Combine all features
    # [a, b, c, alpha, beta, gamma, vol, density, frac_Al, frac_Ga, frac_In, n_atoms]
    features = np.concatenate([cell_specific, [volume], [density], stoich, [n_atoms]])

    return features


def process_geometry(file_path):
    """
    Wrapper to process a single geometry file and return all features.

    Args:
        file_path (str): Relative path to geometry file.

    Returns:
        atomic_features (np.ndarray): (N, Atomic_Dim)
        global_features (np.ndarray): (Global_Dim,)
    """
    # Load Config parameters
    k = Config.K_NEIGHBORS
    atom_types = Config.ATOM_TYPES

    # 1. Parse
    atoms = parse_xyz(file_path)
    n_atoms = len(atoms)

    # 2. Centered Coordinates
    # Center relative to geometric center of atoms
    positions = atoms.get_positions()
    center_of_geometry = np.mean(positions, axis=0)
    centered_coords = positions - center_of_geometry

    # 3. Neighbors and Distances
    d_neighbors, idx_neighbors = get_pbc_neighbors(atoms, k)

    if d_neighbors.shape[1] > 0:
        # d_min: distance to closest neighbor
        d_min = d_neighbors[:, 0:1]
        # d_mean: average distance to k neighbors
        d_mean = np.mean(d_neighbors, axis=1, keepdims=True)
    else:
        d_min = np.zeros((n_atoms, 1))
        d_mean = np.zeros((n_atoms, 1))

    # Local Stoichiometry
    symbols = atoms.get_chemical_symbols()
    local_stoich = compute_local_stoichiometry(idx_neighbors, symbols, atom_types)

    # 4. Atomic Identity (One-hot)
    type_map = {sym: i for i, sym in enumerate(atom_types)}
    one_hot = np.zeros((n_atoms, len(atom_types)))
    for i, sym in enumerate(symbols):
        if sym in type_map:
            one_hot[i, type_map[sym]] = 1.0

    # 5. Assemble Atomic Features
    # [One-hot(4), Coords(3), d_min(1), d_mean(1), LocalStoich(4)] -> 13 dims
    atomic_features = np.concatenate(
        [one_hot, centered_coords, d_min, d_mean, local_stoich], axis=1
    )

    # 6. Global Features
    global_feat = compute_global_features(atoms, atom_types)

    return atomic_features.astype(np.float32), global_feat.astype(np.float32)
