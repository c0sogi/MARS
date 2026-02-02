import os
import numpy as np
import torch
from ase.io import read
from library.config import Config


def load_molecule(file_path):
    """
    Parses an xyz file using ASE.

    Args:
        file_path (str): Relative path to the geometry.xyz file (e.g., "train/1/geometry.xyz").

    Returns:
        ase.Atoms: An ASE Atoms object containing atomic positions, numbers, and cell info.
    """
    # Construct full path using the input directory defined in Config
    full_path = os.path.join(Config.INPUT_DIR, file_path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Geometry file not found at: {full_path}")

    # ASE read handles the format automatically.
    # The provided files use a standard format compatible with ASE.
    try:
        # Cite debug_lesson_8: Explicitly Override File Format Inference for Ambiguous Extensions
        atoms = read(full_path, format="aims")
        return atoms
    except Exception as e:
        raise ValueError(f"Failed to parse molecule file {full_path}: {e}")


def get_pbc_distances(atoms):
    """
    Computes the pairwise distance matrix respecting periodic boundary conditions (MIC).

    Args:
        atoms (ase.Atoms): The atoms object containing positions and cell information.

    Returns:
        np.ndarray: A square matrix of distances (N_atoms x N_atoms) where D_ij is the
                    distance between atom i and the nearest periodic image of atom j.
    """
    # get_all_distances with mic=True applies the Minimum Image Convention (MIC)
    # This ensures that we calculate the shortest distance in the periodic lattice.
    distances = atoms.get_all_distances(mic=True)
    return distances


def calculate_cell_volume(lattice_lengths, lattice_angles):
    """
    Calculates unit cell volume from lattice lengths and angles for a triclinic cell.

    Args:
        lattice_lengths (list or np.ndarray): [a, b, c] lengths in Angstroms.
        lattice_angles (list or np.ndarray): [alpha, beta, gamma] angles in degrees.

    Returns:
        float: The volume of the unit cell in Angstrom^3.
    """
    a, b, c = lattice_lengths
    alpha, beta, gamma = lattice_angles

    # Convert angles from degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Formula for the volume of a general triclinic unit cell
    # V = abc * sqrt(1 - cos^2(alpha) - cos^2(beta) - cos^2(gamma) + 2cos(alpha)cos(beta)cos(gamma))
    term = (
        1
        - np.cos(alpha_rad) ** 2
        - np.cos(beta_rad) ** 2
        - np.cos(gamma_rad) ** 2
        + 2 * np.cos(alpha_rad) * np.cos(beta_rad) * np.cos(gamma_rad)
    )

    # Use max(0, term) to prevent numerical errors (NaN) if term is slightly negative due to float precision
    volume = a * b * c * np.sqrt(max(0.0, term))
    return volume


def inverse_transform_targets(y_pred):
    """
    Converts model predictions back to the original scale.
    The training transformation is log(1 + y).
    The inverse transformation is exp(y) - 1.

    Args:
        y_pred (np.ndarray or torch.Tensor): Predicted values in the log-transformed scale.

    Returns:
        np.ndarray: Predictions in the original energy scale.
    """
    # Convert torch Tensor to numpy array if necessary
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Apply expm1 (exp(x) - 1) which is the inverse of log1p
    return np.expm1(y_pred)
