import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_mcrmse(preds, targets):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the scored targets.

    This function handles:
    1. Slicing predictions to the scored sequence length (first 68 bases).
    2. Filtering for the specific scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    3. Computing RMSE globally for each column and then averaging.

    Args:
        preds (np.ndarray or torch.Tensor): Predictions of shape (N, seq_len, 5).
                                            Typically seq_len is 107.
        targets (np.ndarray or torch.Tensor): Ground truth of shape (N, seq_scored, 5).
                                              Typically seq_scored is 68.

    Returns:
        float: The calculated MCRMSE score.
        dict: A dictionary mapping column names to their individual RMSE scores.
    """
    # Convert Tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Slice predictions to the scored length (e.g., 68)
    # Targets are expected to be already of length 68, but we slice just in case
    # to ensure shape compatibility if full-length targets are passed.
    preds_sliced = preds[:, : Config.SEQ_SCORED, :]
    targets_sliced = targets[:, : Config.SEQ_SCORED, :]

    # Identify indices of the columns that contribute to the score
    # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_TARGETS
    ]

    # Filter arrays to keep only the scored columns
    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets_sliced[:, :, scored_indices]

    # Calculate Squared Error: (y - y_hat)^2
    squared_error = (preds_filtered - targets_filtered) ** 2

    # Calculate Mean Squared Error per column
    # Average over samples (axis 0) and sequence positions (axis 1)
    mse_per_column = np.mean(squared_error, axis=(0, 1))

    # Calculate RMSE per column
    rmse_per_column = np.sqrt(mse_per_column)

    # Calculate MCRMSE (Mean of the column RMSEs)
    mcrmse = np.mean(rmse_per_column)

    # Create a dictionary for detailed logging
    col_scores = {
        Config.TARGET_COLS[original_idx]: score
        for original_idx, score in zip(scored_indices, rmse_per_column)
    }

    return mcrmse, col_scores
