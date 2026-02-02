import torch
import numpy as np
from library.config import set_seed


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    This function wraps the set_seed function from the library configuration
    to ensure consistent behavior.

    Args:
        seed (int): The seed value to be set.
    """
    set_seed(seed)


def mcrmse_metric(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is computed as the average of the RMSE values calculated
    independently for each of the scored target columns (reactivity, deg_Mg_pH10, deg_Mg_50C).

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth target values.
            Expected shape: (Batch_Size, Seq_Len, Num_Targets) or (N_Samples, Num_Targets).
        y_pred (np.ndarray or torch.Tensor): Predicted target values.
            Must have the same shape as y_true.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Detach and move to CPU if inputs are torch Tensors
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are float32 for consistent precision
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    # Calculate squared errors
    squared_diff = (y_true - y_pred) ** 2

    # Compute MSE per column
    # If 3D (Batch, Seq, Channels), average over batch (0) and sequence (1)
    if y_true.ndim == 3:
        mse_per_column = np.mean(squared_diff, axis=(0, 1))
    # If 2D (N_samples, Channels), average over samples (0)
    elif y_true.ndim == 2:
        mse_per_column = np.mean(squared_diff, axis=0)
    else:
        raise ValueError(f"Input arrays must be 2D or 3D, got shape {y_true.shape}")

    # Compute RMSE per column
    rmse_per_column = np.sqrt(mse_per_column)

    # Compute the mean of the column-wise RMSEs
    mcrmse = np.mean(rmse_per_column)

    return float(mcrmse)
