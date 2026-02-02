import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Enforce deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_structure_adj(structure_str):
    """
    Parses a dot-bracket structure string into an adjacency index array.

    This function is used to generate the gathering indices for the Decoupled
    Structural Interaction Module.

    Args:
        structure_str (str): A string representing RNA secondary structure
                             (e.g., "((..))").

    Returns:
        np.ndarray: A 1D numpy array of integers with length equal to the sequence.
                    If position i is paired with j, arr[i] = j.
                    If position i is unpaired, arr[i] = -1.
    """
    seq_len = len(structure_str)
    # Initialize with -1 (unpaired)
    adj = np.full(seq_len, -1, dtype=np.int32)
    stack = []

    for i, char in enumerate(structure_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            if stack:
                j = stack.pop()
                # Record the pair in both directions
                adj[i] = j
                adj[j] = i

    return adj


def scored_mcrmse(y_pred, y_true):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the
    scored targets.

    This function strictly follows the competition evaluation protocol:
    1. Slices data to the first `Config.SEQ_SCORED` positions (68).
    2. Filters for `Config.SCORED_COLS` (reactivity, deg_Mg_pH10, deg_Mg_50C).
    3. Computes RMSE for each column globally (or over the provided batch).
    4. Returns the mean of these RMSE values.

    Args:
        y_pred (torch.Tensor): Predicted values of shape (N, L, 5).
        y_true (torch.Tensor): Ground truth values of shape (N, L, 5) or (N, 68, 5).

    Returns:
        torch.Tensor: The scalar MCRMSE score.
    """
    # Determine indices of the columns that count towards the score
    # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # 1. Slice to the scored sequence length (first 68 bases)
    seq_scored = Config.SEQ_SCORED

    # Handle cases where y_true might already be sliced or is full length
    # We slice both to ensure consistency up to seq_scored
    pred_sliced = y_pred[:, :seq_scored, :]
    true_sliced = y_true[:, :seq_scored, :]

    # 2. Select only the scored columns
    pred_filtered = pred_sliced[:, :, scored_indices]
    true_filtered = true_sliced[:, :, scored_indices]

    # 3. Compute MSE per column
    # Average over Batch (dim 0) and Sequence (dim 1) dimensions
    mse_per_col = torch.mean((pred_filtered - true_filtered) ** 2, dim=(0, 1))

    # 4. Compute RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)

    # 5. Average RMSE across the scored columns
    mcrmse = torch.mean(rmse_per_col)

    return mcrmse
