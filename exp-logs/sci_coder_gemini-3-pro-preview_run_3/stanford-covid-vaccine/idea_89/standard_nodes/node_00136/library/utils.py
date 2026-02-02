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


def get_scored_indices():
    """
    Retrieves the indices of the scored targets within the full target list based on Config.

    Returns:
        list: Indices of Config.SCORED_TARGETS within Config.TARGET_COLS.
    """
    return [Config.TARGET_COLS.index(col) for col in Config.SCORED_TARGETS]


def mcrmse_metric(y_true, y_pred, seq_scored=Config.SEQ_SCORED, target_indices=None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric slices the input sequences to the scored length (seq_scored)
    and computes the RMSE for each column, then averages the RMSEs.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values. Shape (N, SeqLen, NumTargets).
        y_pred (torch.Tensor or np.ndarray): Predicted values. Shape (N, SeqLen, NumTargets).
        seq_scored (int): The number of sequence positions to include in the score (starting from 0).
                          Defaults to Config.SEQ_SCORED.
        target_indices (list, optional): List of column indices to include in the final metric.
                                         If None, all columns are used.

    Returns:
        float: The calculated MCRMSE value.
    """
    # Convert numpy arrays to torch tensors if necessary
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Ensure tensors are on CPU to prevent unnecessary GPU memory usage during metric calc
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    # Slice the sequence dimension to include only scored positions
    # Assumes shape is (Batch, SeqLen, Channels)
    if y_true.shape[1] >= seq_scored:
        y_true = y_true[:, :seq_scored, :]
        y_pred = y_pred[:, :seq_scored, :]

    # Calculate Squared Error: (y - y_hat)^2
    squared_error = (y_true - y_pred) ** 2

    # Calculate Mean Squared Error (MSE) per column
    # Average over Batch (dim 0) and Sequence (dim 1)
    mse_per_column = torch.mean(squared_error, dim=(0, 1))

    # Calculate Root Mean Squared Error (RMSE) per column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Filter for specific target columns if indices are provided
    if target_indices is not None:
        rmse_per_column = rmse_per_column[target_indices]

    # Calculate Mean of RMSEs (MCRMSE)
    mcrmse = torch.mean(rmse_per_column)

    return mcrmse.item()
