import os
import numpy as np
import ase.io
from ase import neighborlist
from library.config import INPUT_DIR


def load_xyz(file_path: str) -> ase.Atoms:
    """
    Reads a geometry.xyz file and returns an ASE Atoms object.

    Args:
        file_path (str): Relative path (e.g., 'train/1/geometry.xyz') or absolute path.

    Returns:
        ase.Atoms: The crystal structure object.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If parsing fails.
    """
    # Handle path resolution
    if not os.path.isabs(file_path) and not file_path.startswith(INPUT_DIR):
        full_path = os.path.join(INPUT_DIR, file_path)
    else:
        full_path = file_path

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Structure file not found at: {full_path}")

    try:
        # ASE handles the parsing of the .xyz format including lattice vectors
        atoms = ase.io.read(full_path)
        return atoms
    except Exception as e:
        raise RuntimeError(f"Failed to parse structure file {full_path}: {e}")


def get_neighbor_list(
    atoms: ase.Atoms, cutoff: float = 5.0, self_interaction: bool = False
):
    """
    Computes the neighbor list for an ASE Atoms object using a radial cutoff.
    Returns indices, distances, and displacement vectors.

    Args:
        atoms (ase.Atoms): The crystal structure.
        cutoff (float): The radial cutoff distance in Angstroms.
        self_interaction (bool): Whether to include self-loops (atom i interacting with itself).

    Returns:
        tuple: (idx_i, idx_j, distances, displacement_vectors)
            - idx_i (np.ndarray): Indices of the central atoms.
            - idx_j (np.ndarray): Indices of the neighboring atoms.
            - distances (np.ndarray): Scalar distances between i and j.
            - displacement_vectors (np.ndarray): Vectors pointing from i to j (j - i).
    """
    # 'i': index of first atom
    # 'j': index of second atom
    # 'd': distance between atoms
    # 'D': vector pointing from i to j
    i, j, d, D = neighborlist.neighbor_list(
        "ijdD", atoms, cutoff, self_interaction=self_interaction
    )

    return i, j, d, D


def get_cell_parameters(atoms: ase.Atoms) -> dict:
    """
    Extracts lattice parameters (lengths and angles) and volume.

    Args:
        atoms (ase.Atoms): The crystal structure.

    Returns:
        dict: Dictionary containing 'a', 'b', 'c', 'alpha', 'beta', 'gamma', and 'volume'.
    """
    cell = atoms.get_cell()
    lengths_angles = cell.cellpar()  # [a, b, c, alpha, beta, gamma]
    volume = cell.volume

    return {
        "a": lengths_angles[0],
        "b": lengths_angles[1],
        "c": lengths_angles[2],
        "alpha": lengths_angles[3],
        "beta": lengths_angles[4],
        "gamma": lengths_angles[5],
        "volume": volume,
    }
