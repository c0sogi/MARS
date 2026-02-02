import random
import numpy as np
import torch
import itertools
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_pbc_distance(coords, lattice):
    """
    Computes the nearest neighbor distance for each atom in the unit cell,
    respecting Periodic Boundary Conditions (PBC).

    This function generates periodic images of the unit cell to find the
    true closest neighbor for every atom, which is critical for accurate
    local geometric feature extraction in crystalline materials.

    Args:
        coords (np.ndarray): Atomic coordinates of shape (N, 3).
        lattice (np.ndarray): Lattice vectors of shape (3, 3).

    Returns:
        np.ndarray: Nearest neighbor distance for each atom, shape (N,).
    """
    # Generate 27 periodic images (shifts) corresponding to indices {-1, 0, 1}
    ranges = [-1, 0, 1]
    indices = np.array(
        list(itertools.product(ranges, ranges, ranges))
    )  # Shape: (27, 3)

    # Calculate shift vectors in Cartesian coordinates
    shifts = indices @ lattice  # Shape: (27, 3)

    # Compute pairwise difference vectors between all atoms in the central cell
    # coords[:, None, :] is (N, 1, 3)
    # coords[None, :, :] is (1, N, 3)
    # diff_vectors is (N, N, 3) representing r_i - r_j
    diff_vectors = coords[:, None, :] - coords[None, :, :]

    # Add lattice shifts to difference vectors
    # Broadcasting: (N, N, 1, 3) + (1, 1, 27, 3) -> (N, N, 27, 3)
    all_diffs = diff_vectors[:, :, None, :] + shifts[None, None, :, :]

    # Compute Euclidean distances for all pairs and images
    # Shape: (N, N, 27)
    dists = np.linalg.norm(all_diffs, axis=-1)

    # Mask self-distance (where distance is effectively 0)
    # We use a small epsilon to avoid numerical noise issues.
    # This handles the i=j case in the central image (shift [0,0,0]).
    dists[dists < 1e-6] = np.inf

    # Find the minimum distance for each atom i across all j and all periodic images
    # Shape: (N,)
    nn_dists = np.min(dists, axis=(1, 2))

    return nn_dists


class MetricTracker:
    """
    Computes and stores the running average and current value of a metric.
    Useful for tracking loss and evaluation metrics during training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the running average.

        Args:
            val (float): The current value to update (e.g., batch loss).
            n (int): The number of samples associated with val (e.g., batch size).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Returns the current average formatted as a string."""
        return f"{self.avg:.6f}"
