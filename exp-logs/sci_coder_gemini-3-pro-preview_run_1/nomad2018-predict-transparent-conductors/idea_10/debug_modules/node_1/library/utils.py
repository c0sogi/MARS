import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to set.
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


def calculate_cell_volume(lattice_matrix: np.ndarray) -> float:
    """
    Calculates the volume of the unit cell given the lattice matrix.

    Args:
        lattice_matrix (np.ndarray): (3, 3) matrix where rows are lattice vectors.

    Returns:
        float: The volume of the unit cell.
    """
    return np.abs(np.linalg.det(lattice_matrix))


def calculate_lattice_params(lattice_matrix: np.ndarray):
    """
    Calculates lattice lengths (a, b, c) and angles (alpha, beta, gamma) from the lattice matrix.

    Args:
        lattice_matrix (np.ndarray): (3, 3) matrix where rows are lattice vectors.

    Returns:
        tuple: (lengths, angles) where both are np.ndarray of shape (3,).
               lengths = [a, b, c]
               angles = [alpha, beta, gamma] in degrees.
    """
    # Lattice vectors are rows
    a_vec = lattice_matrix[0]
    b_vec = lattice_matrix[1]
    c_vec = lattice_matrix[2]

    # Lengths
    a = np.linalg.norm(a_vec)
    b = np.linalg.norm(b_vec)
    c = np.linalg.norm(c_vec)

    # Angles
    # alpha: angle between b and c
    # beta: angle between a and c
    # gamma: angle between a and b
    # Clip values to [-1, 1] to handle potential floating point errors
    alpha_rad = np.arccos(np.clip(np.dot(b_vec, c_vec) / (b * c), -1.0, 1.0))
    beta_rad = np.arccos(np.clip(np.dot(a_vec, c_vec) / (a * c), -1.0, 1.0))
    gamma_rad = np.arccos(np.clip(np.dot(a_vec, b_vec) / (a * b), -1.0, 1.0))

    lengths = np.array([a, b, c])
    angles = np.degrees(np.array([alpha_rad, beta_rad, gamma_rad]))

    return lengths, angles


def get_pbc_distances(positions: np.ndarray, lattice_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates pairwise distances between atoms respecting periodic boundary conditions
    (Minimum Image Convention) by checking all 27 nearest cell images.

    Args:
        positions (np.ndarray): (N, 3) Cartesian coordinates of atoms.
        lattice_matrix (np.ndarray): (3, 3) Lattice vectors as rows.

    Returns:
        np.ndarray: (N, N) Distance matrix where D[i, j] is the minimum distance
                    between atom i and atom j (or its periodic image).
    """
    # Generate 27 translation vectors (3x3x3 neighborhood)
    # shifts indices: -1, 0, 1
    ranges = [-1, 0, 1]
    shifts = []
    for i in ranges:
        for j in ranges:
            for k in ranges:
                shift = (
                    i * lattice_matrix[0]
                    + j * lattice_matrix[1]
                    + k * lattice_matrix[2]
                )
                shifts.append(shift)
    shifts = np.array(shifts)  # Shape: (27, 3)

    # Expand dimensions for broadcasting
    # pos_i: (N, 1, 1, 3) -> positions of atom i
    # pos_j: (1, N, 1, 3) -> positions of atom j
    # shifts_br: (1, 1, 27, 3) -> all possible cell shifts

    pos_i = positions[:, np.newaxis, np.newaxis, :]
    pos_j = positions[np.newaxis, :, np.newaxis, :]
    shifts_br = shifts[np.newaxis, np.newaxis, :, :]

    # Calculate difference vectors for all pairs and all shifts
    # diff_vectors = r_i - (r_j + shift)
    diff_vectors = pos_i - (pos_j + shifts_br)  # Shape: (N, N, 27, 3)

    # Calculate squared distances
    dist_sq = np.sum(diff_vectors**2, axis=-1)  # Shape: (N, N, 27)

    # Find minimum squared distance across all 27 images for each pair
    min_dist_sq = np.min(dist_sq, axis=2)  # Shape: (N, N)

    return np.sqrt(min_dist_sq)


def rmsle(y_true, y_pred):
    """
    Calculates the mean Column-wise Root Mean Squared Logarithmic Error.

    Args:
        y_true: Ground truth values (numpy array or torch tensor).
        y_pred: Predicted values (numpy array or torch tensor).

    Returns:
        float: The mean RMSLE over all target columns.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure no negative values for log (clip to 0)
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)

    # Calculate squared log error
    squared_log_error = (np.log1p(y_pred) - np.log1p(y_true)) ** 2

    # Mean over samples (axis 0) to get MSE per column, then sqrt to get RMSE per column
    column_rmsle = np.sqrt(np.mean(squared_log_error, axis=0))

    # Return mean over columns (scalar)
    return np.mean(column_rmsle)
