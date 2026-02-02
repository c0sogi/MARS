import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python's random, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_scored_indices():
    """
    Retrieves the indices of the columns that are used for the competition scoring.

    Returns:
        list: A list of integer indices corresponding to the scored columns.
    """
    return [i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS]


def mcrmse(y_true, y_pred, only_scored=True):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function computes the RMSE for each target column separately and then averages them.
    It automatically slices the predictions to match the length of the ground truth (typically 68)
    and can filter to only include the specific columns used for the competition leaderboard.

    Args:
        y_true (torch.Tensor): Ground truth values. Shape (Batch, Seq_Len_True, Num_Targets).
                               Typically (Batch, 68, 5).
        y_pred (torch.Tensor): Predicted values. Shape (Batch, Seq_Len_Pred, Num_Targets).
                               Typically (Batch, 107, 5).
        only_scored (bool): If True, the metric is calculated only on the scored columns
                            (reactivity, deg_Mg_pH10, deg_Mg_50C). If False, it is calculated
                            on all 5 columns. Defaults to True.

    Returns:
        torch.Tensor: A scalar tensor representing the MCRMSE.
    """
    # Ensure inputs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)

    # Slice predictions to match the scored sequence length (Config.PRED_LEN = 68)
    # y_true is expected to be of length 68 in the sequence dimension.
    seq_len_scored = y_true.shape[1]

    # Slice y_pred if it is longer than y_true (e.g., 107 vs 68)
    if y_pred.shape[1] > seq_len_scored:
        y_pred = y_pred[:, :seq_len_scored, :]
    elif y_pred.shape[1] < seq_len_scored:
        # Fallback/Safety: Slice y_true to match y_pred if y_pred is unexpectedly shorter
        y_true = y_true[:, : y_pred.shape[1], :]

    # Calculate Squared Error
    squared_error = (y_true - y_pred) ** 2

    # Calculate Mean Squared Error per column
    # Average over batch (dim 0) and sequence (dim 1)
    mse_per_column = torch.mean(squared_error, dim=(0, 1))

    # Calculate RMSE per column
    rmse_per_column = torch.sqrt(mse_per_column)

    if only_scored:
        # Filter for the scored columns
        scored_indices = get_scored_indices()
        # Create indices tensor on the same device as the data
        indices = torch.tensor(scored_indices, device=rmse_per_column.device)
        rmse_per_column = torch.index_select(rmse_per_column, 0, indices)

    # Return the mean of the RMSEs
    return torch.mean(rmse_per_column)
