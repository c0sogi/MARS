import os
import numpy as np
from ase import Atoms
from ase.io import read
from library.config import INPUT_DIR


def load_atoms(relative_path: str) -> Atoms:
    """
    Parses a geometry file into an ASE Atoms object.

    The dataset files have a .xyz extension but the content follows the FHI-aims
    geometry format (using 'lattice_vector' and 'atom' keywords). This function
    handles that discrepancy and ensures Periodic Boundary Conditions (PBC) are enabled.

    Args:
        relative_path: Path to the geometry file relative to the input directory
                      (e.g., 'train/1/geometry.xyz').

    Returns:
        ase.Atoms: The loaded atomic structure with PBC enabled.

    Raises:
        FileNotFoundError: If the file does not exist at the constructed path.
    """
    full_path = os.path.join(INPUT_DIR, relative_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found at: {full_path}")

    # Explicitly use 'aims' format because the content uses FHI-aims syntax
    # despite the .xyz extension.
    atoms = read(full_path, format="aims")

    # Ensure periodic boundary conditions are set to True for all 3 dimensions
    # as these are bulk crystal structures.
    atoms.set_pbc(True)

    return atoms


def get_pbc_distances(atoms: Atoms) -> np.ndarray:
    """
    Computes the full pairwise distance matrix respecting periodic boundary conditions.

    This uses the Minimum Image Convention (MIC) to find the shortest distance
    between any two atoms in the periodic lattice.

    Args:
        atoms: ASE Atoms object.

    Returns:
        np.ndarray: A square matrix of shape (N, N) where N is the number of atoms.
                    Entry [i, j] is the distance between atom i and atom j under PBC.
    """
    # mic=True enables the Minimum Image Convention
    return atoms.get_all_distances(mic=True)


def get_centered_positions(atoms: Atoms) -> np.ndarray:
    """
    Centers atomic coordinates relative to the unit cell centroid.

    This is useful for creating translation-invariant features for neural networks
    by removing the absolute spatial offset of the structure.

    Args:
        atoms: ASE Atoms object.

    Returns:
        np.ndarray: Array of shape (N, 3) containing the centered Cartesian coordinates.
    """
    positions = atoms.get_positions()
    centroid = np.mean(positions, axis=0)
    return positions - centroid
