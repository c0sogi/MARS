import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metric_mcrmse(
    y_true: np.ndarray, y_pred: np.ndarray, seq_scored: int = 68
) -> float:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the scored positions and columns.

    The metric is calculated as follows:
    1. Slice the data to the first `seq_scored` positions (typically 68).
    2. Select the specific columns used for scoring: reactivity, deg_Mg_pH10, and deg_Mg_50C.
       Assumes input shape is (N, seq_len, 5) with column order:
       [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
    3. Compute RMSE for each selected column.
    4. Return the mean of these RMSE values.

    Args:
        y_true (np.ndarray): Ground truth values of shape (N, seq_len, 5).
        y_pred (np.ndarray): Predicted values of shape (N, seq_len, 5).
        seq_scored (int): The number of sequence positions to score (default 68).

    Returns:
        float: The MCRMSE score.
    """
    # 1. Slice to scored sequence length
    # Shape becomes (N, 68, 5)
    targs = y_true[:, :seq_scored, :]
    preds = y_pred[:, :seq_scored, :]

    # 2. Select scored columns
    # Indices: 0=reactivity, 1=deg_Mg_pH10, 3=deg_Mg_50C
    # We ignore 2=deg_pH10 and 4=deg_50C for the metric
    scored_indices = [0, 1, 3]

    targs_scored = targs[:, :, scored_indices]
    preds_scored = preds[:, :, scored_indices]

    # 3. Compute RMSE per column
    # Calculate squared error
    squared_error = (targs_scored - preds_scored) ** 2

    # Mean over samples (axis 0) and sequence positions (axis 1)
    # Result is a vector of MSEs, one per scored column
    mse_per_col = np.mean(squared_error, axis=(0, 1))

    # RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # 4. Mean of RMSEs
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
