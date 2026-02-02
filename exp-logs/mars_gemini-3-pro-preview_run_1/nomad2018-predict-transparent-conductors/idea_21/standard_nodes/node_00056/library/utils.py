import os
import random
import numpy as np
import torch
from ase.geometry import get_distances


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pbc_distance_matrix(
    positions: np.ndarray, lattice_vectors: np.ndarray
) -> np.ndarray:
    """
    Calculates the pairwise distance matrix for atoms in a unit cell,
    respecting periodic boundary conditions (PBC).

    Args:
        positions (np.ndarray): Atomic positions of shape (N, 3).
        lattice_vectors (np.ndarray): Lattice vectors of shape (3, 3).

    Returns:
        np.ndarray: Distance matrix of shape (N, N).
    """
    # Use ASE's get_distances to handle PBC correctly for general triclinic cells
    # p1=positions: The atoms to calculate distances between
    # cell=lattice_vectors: The unit cell definition
    # pbc=True: Enable periodic boundary conditions in all directions

    # get_distances returns a tuple: (displacement_vectors, distance_matrix)
    # We only need the distance_matrix (index 1)
    _, distance_matrix = get_distances(p1=positions, cell=lattice_vectors, pbc=True)

    return distance_matrix


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Column-wise Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_true (np.ndarray): Ground truth values of shape (N, D).
        y_pred (np.ndarray): Predicted values of shape (N, D).

    Returns:
        float: The mean RMSLE across all target columns.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to be non-negative to avoid log domain errors
    # RMSLE is defined for non-negative values
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)

    # Calculate squared logarithmic errors
    # log1p(x) = log(1 + x)
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    squared_log_errors = np.square(log_diff)

    # Calculate Mean Squared Log Error (MSLE) for each column
    msle_per_column = np.mean(squared_log_errors, axis=0)

    # Calculate Root Mean Squared Log Error (RMSLE) for each column
    rmsle_per_column = np.sqrt(msle_per_column)

    # Return the average RMSLE across columns (Column-wise RMSLE)
    return np.mean(rmsle_per_column)
