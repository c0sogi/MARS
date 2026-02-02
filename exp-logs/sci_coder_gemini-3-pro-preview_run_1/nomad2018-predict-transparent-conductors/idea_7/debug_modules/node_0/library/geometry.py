import numpy as np
from ase import Atoms


def read_xyz(file_path):
    """
    Parses the custom .xyz file format provided in the dataset.

    Args:
        file_path (str): Path to the geometry.xyz file.

    Returns:
        tuple: (lattice, atom_types, atom_coords)
            - lattice: 3x3 numpy array of lattice vectors.
            - atom_types: List of strings representing element symbols.
            - atom_coords: Nx3 numpy array of atomic Cartesian coordinates.
    """
    lattice = []
    atom_types = []
    atom_coords = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if parts[0] == "lattice_vector":
                lattice.append([float(x) for x in parts[1:4]])
            elif parts[0] == "atom":
                atom_coords.append([float(x) for x in parts[1:4]])
                atom_types.append(parts[4])

    return np.array(lattice), atom_types, np.array(atom_coords)


def center_coordinates(coords, lattice):
    """
    Shifts atomic positions to center them around the unit cell centroid.

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.
        lattice (np.ndarray): 3x3 array of lattice vectors.

    Returns:
        np.ndarray: Centered coordinates.
    """
    # Calculate the centroid of the unit cell (parallelepiped)
    # Centroid is the midpoint of the body diagonal: 0.5 * (v1 + v2 + v3)
    cell_centroid = np.sum(lattice, axis=0) * 0.5

    # Subtract the centroid from all atomic coordinates
    centered_coords = coords - cell_centroid

    return centered_coords


def compute_pbc_neighbor_distances(coords, lattice):
    """
    Calculates the minimum pairwise distance for each atom while strictly
    respecting periodic boundary conditions defined by the lattice vectors.

    Args:
        coords (np.ndarray): Nx3 array of atomic coordinates.
        lattice (np.ndarray): 3x3 array of lattice vectors.

    Returns:
        np.ndarray: Array of shape (N,) containing the distance to the nearest neighbor for each atom.
    """
    # Create an ASE Atoms object to leverage its robust PBC distance calculation
    atoms = Atoms(positions=coords, cell=lattice, pbc=True)

    # Get the full distance matrix using Minimum Image Convention (MIC)
    # This handles the periodic boundaries correctly.
    distance_matrix = atoms.get_all_distances(mic=True)

    # The distance from an atom to itself is 0.0. We need to ignore this
    # to find the nearest *other* neighbor.
    # We replace the diagonal with infinity.
    np.fill_diagonal(distance_matrix, np.inf)

    # Find the minimum distance in each row
    min_distances = np.min(distance_matrix, axis=1)

    return min_distances
