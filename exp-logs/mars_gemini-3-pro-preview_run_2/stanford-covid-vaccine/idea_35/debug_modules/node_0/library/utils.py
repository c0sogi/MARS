import os
import random
import numpy as np
import torch
from library.config import SCORED_INDICES


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def MCRMSE(y_true, y_pred, scored_indices=None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the average of the RMSE values for each scored target column.

    Args:
        y_true (np.ndarray): Ground truth array. Expected shape is (N_samples, seq_len, N_targets)
                             or (N_samples * seq_len, N_targets).
        y_pred (np.ndarray): Predicted array. Must have the same shape as y_true.
        scored_indices (list, optional): List of integer indices indicating which columns
                                         in the last dimension are used for scoring.
                                         If None, all columns are used.

    Returns:
        float: The MCRMSE score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch for MCRMSE: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # If specific columns are specified for scoring, select them
    if scored_indices is not None:
        y_true = np.take(y_true, scored_indices, axis=-1)
        y_pred = np.take(y_pred, scored_indices, axis=-1)

    # Calculate Squared Error
    squared_error = np.square(y_true - y_pred)

    # Calculate MSE per column
    # We average over all dimensions except the last one (which represents the targets)
    reduction_axes = tuple(range(y_true.ndim - 1))
    mse = np.mean(squared_error, axis=reduction_axes)

    # Calculate RMSE per column
    rmse = np.sqrt(mse)

    # Calculate the mean of the RMSEs (MCRMSE)
    mcrmse_val = np.mean(rmse)

    return float(mcrmse_val)
