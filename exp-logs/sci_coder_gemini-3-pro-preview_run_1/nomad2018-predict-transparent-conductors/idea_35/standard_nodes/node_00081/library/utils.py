import os
import numpy as np
from ase import Atoms
from scipy.spatial.distance import cdist
from library.config import Config


def parse_xyz(file_path):
    """
    Parses the custom geometry.xyz file format to create an ASE Atoms object.

    Args:
        file_path (str): Path to the geometry.xyz file.

    Returns:
        ase.Atoms: The atomic structure with periodic boundary conditions.
    """
    lattice_vectors = []
    positions = []
    symbols = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == "lattice_vector":
            # Format: lattice_vector x y z
            vec = [float(x) for x in parts[1:4]]
            lattice_vectors.append(vec)
        elif parts[0] == "atom":
            # Format: atom x y z Symbol
            pos = [float(x) for x in parts[1:4]]
            sym = parts[4]
            positions.append(pos)
            symbols.append(sym)

    # Create ASE Atoms object
    # Assuming full periodicity (pbc=True) as it is a crystal structure task
    atoms = Atoms(symbols=symbols, positions=positions, cell=lattice_vectors, pbc=True)
    return atoms


def compute_pbc_distances(atoms):
    """
    Computes pairwise distances for each atom to all atoms in a 3x3x3 supercell
    to accurately capture nearest neighbors under periodic boundary conditions.

    Args:
        atoms (ase.Atoms): The unit cell structure.

    Returns:
        numpy.ndarray: A sorted distance matrix of shape (N_atoms, N_supercell_atoms).
                       Each row corresponds to an atom in the unit cell and contains
                       distances to all atoms in the immediate periodic neighborhood, sorted ascending.
    """
    # Get unit cell parameters and positions
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    n_atoms = len(atoms)

    # Generate translation vectors for a 3x3x3 supercell (indices -1, 0, 1)
    # This ensures we cover all immediate periodic images
    ranges = [-1, 0, 1]
    offsets = np.array([[i, j, k] for i in ranges for j in ranges for k in ranges])
    # offsets shape: (27, 3)

    # Calculate cartesian translations: (27, 3) @ (3, 3) -> (27, 3)
    translations = offsets @ cell

    # Create supercell positions
    # We broadcast positions (N, 3) with translations (27, 3)
    # Result should be (N*27, 3)
    # positions[:, None, :] shape is (N, 1, 3)
    # translations[None, :, :] shape is (1, 27, 3)
    # Sum is (N, 27, 3), reshape to (N*27, 3)
    supercell_positions = (positions[:, None, :] + translations[None, :, :]).reshape(
        -1, 3
    )

    # Compute distance matrix between unit cell atoms and supercell atoms
    # Shape: (N, N*27)
    dists = cdist(positions, supercell_positions)

    # Sort distances for each atom row
    sorted_dists = np.sort(dists, axis=1)

    return sorted_dists


def get_multi_order_neighbors(sorted_distances, k=3):
    """
    Extracts the k nearest neighbor distances for each atom from the sorted distance matrix.
    Skips the first column which corresponds to the self-distance (0.0).

    Args:
        sorted_distances (numpy.ndarray): Sorted distance matrix from compute_pbc_distances.
        k (int): Number of neighbors to extract (default 3 for d1, d2, d3).

    Returns:
        numpy.ndarray: Array of shape (N_atoms, k) containing the 1st to kth nearest neighbor distances.
    """
    # Handle empty case to preserve feature dimensions
    if sorted_distances.shape[0] == 0:
        return np.zeros((0, k))

    # The 0-th column is the distance to self (0.0).
    # We take columns 1 to k+1.
    # Check if we have enough neighbors (supercell generation guarantees this for N>=1)
    if sorted_distances.shape[1] <= k:
        # Fallback for extremely weird cases, though unlikely with 3x3x3 supercell
        return sorted_distances[:, 1:]

    return sorted_distances[:, 1 : k + 1]


def calculate_apf(atoms):
    """
    Calculates the Atomic Packing Factor (APF) using the covalent radii defined in Config.
    APF = (Sum of atomic volumes) / Unit Cell Volume

    Args:
        atoms (ase.Atoms): The atomic structure.

    Returns:
        float: The atomic packing factor.
    """
    # Get chemical symbols
    symbols = atoms.get_chemical_symbols()

    # Calculate total atomic volume
    total_atomic_volume = 0.0
    for sym in symbols:
        radius = Config.COVALENT_RADII.get(
            sym, 1.0
        )  # Default to 1.0 if unknown (should not happen)
        vol = (4.0 / 3.0) * np.pi * (radius**3)
        total_atomic_volume += vol

    # Get unit cell volume
    cell_volume = atoms.get_volume()

    # Avoid division by zero
    if cell_volume < 1e-6:
        return 0.0

    return total_atomic_volume / cell_volume
