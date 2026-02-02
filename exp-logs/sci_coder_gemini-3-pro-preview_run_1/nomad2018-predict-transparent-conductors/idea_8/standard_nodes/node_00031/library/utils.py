import random
import os
import sys
import logging
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_logger(name, log_file=None):
    """
    Creates a logger that prints to stdout and optionally to a file.

    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicates
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Stream handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


def calculate_pbc_distance(coords, lattice):
    """
    Calculates the minimum distance from each atom to its nearest neighbor,
    respecting periodic boundary conditions (PBC).

    This function considers the 27 nearest periodic images (3x3x3 grid) to find
    the true nearest neighbor in the crystal lattice.

    Args:
        coords (np.ndarray): Atomic coordinates of shape (N, 3).
        lattice (np.ndarray): Lattice vectors of shape (3, 3).

    Returns:
        np.ndarray: Array of shape (N,) containing the nearest neighbor distance for each atom.
    """
    N = coords.shape[0]

    # Generate 27 periodic images translations (3x3x3 grid)
    # Range -1 to 1 for x, y, z
    shifts_idx = np.array(
        [[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]]
    )  # Shape (27, 3)

    # Convert index shifts to cartesian vectors using lattice
    # shifts_idx (27, 3) @ lattice (3, 3) -> (27, 3)
    shifts = shifts_idx @ lattice

    # Compute all pairwise vectors between original atoms
    # diffs[i, j, :] = coords[i] - coords[j]
    # Shape (N, N, 3)
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]

    # Add periodic shifts
    # We want to compare diffs[i, j] + shift[k]
    # Reshape diffs to (N, N, 1, 3) and shifts to (1, 1, 27, 3)
    # Result: (N, N, 27, 3)
    all_diffs = diffs[:, :, np.newaxis, :] + shifts[np.newaxis, np.newaxis, :, :]

    # Compute distances squared
    # Shape (N, N, 27)
    dists_sq = np.sum(all_diffs**2, axis=-1)

    # Flatten the neighbor dimension (N * 27)
    # Shape (N, N*27)
    dists_sq_flat = dists_sq.reshape(N, -1)

    # We need to ignore the distance to self at shift (0,0,0)
    # The self-distance is exactly 0.0 (or very close due to float precision)
    # We use a small epsilon threshold to mask out the self-interaction.
    mask = dists_sq_flat > 1e-8

    # Cite debug_lesson_4: Enforce Feature Dimensions on Empty Arrays
    # Handle empty input case to avoid broadcasting errors
    if N == 0:
        return np.zeros(0)

    # Initialize result array
    min_dists = np.zeros(N)

    for i in range(N):
        # Get distances for atom i
        row = dists_sq_flat[i]
        # Apply mask to filter out self-interaction (distance ~ 0)
        valid_dists = row[mask[i]]

        if valid_dists.size > 0:
            min_dist_sq = np.min(valid_dists)
            min_dists[i] = np.sqrt(min_dist_sq)
        else:
            # This case implies no neighbors found > epsilon, which shouldn't happen
            # in a crystal unless it's a single atom in an infinite void (impossible here).
            min_dists[i] = 0.0

    return min_dists
