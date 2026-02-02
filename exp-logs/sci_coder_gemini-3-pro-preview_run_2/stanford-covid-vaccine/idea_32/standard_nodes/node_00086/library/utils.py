import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_metric(y_true, y_pred, scored_indices=None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Formula: MCRMSE = (1/Nt) * sum_j(sqrt((1/n) * sum_i(y_ij - y_hat_ij)^2))
    where Nt is the number of scored columns, and n is the number of samples.

    Args:
        y_true (np.ndarray): Ground truth values. Can be shape (N, Targets) or (N, L, Targets).
        y_pred (np.ndarray): Predicted values. Must have same shape as y_true.
        scored_indices (list[int], optional): List of column indices to include in the metric.
                                              If None, all columns are used.

    Returns:
        float: The MCRMSE score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Flatten all dimensions except the last one (targets)
    # This ensures we calculate RMSE over all predictions (samples * sequence_positions)
    num_targets = y_true.shape[-1]
    y_true_flat = y_true.reshape(-1, num_targets)
    y_pred_flat = y_pred.reshape(-1, num_targets)

    # Filter for scored columns if indices are provided
    if scored_indices is not None:
        y_true_flat = y_true_flat[:, scored_indices]
        y_pred_flat = y_pred_flat[:, scored_indices]

    # Calculate Mean Squared Error per column
    mse_per_col = np.mean((y_true_flat - y_pred_flat) ** 2, axis=0)

    # Calculate Root Mean Squared Error per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate Mean of RMSEs
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
