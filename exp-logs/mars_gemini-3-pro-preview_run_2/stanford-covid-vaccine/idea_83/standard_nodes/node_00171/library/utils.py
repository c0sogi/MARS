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
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_global_rmse(y_pred, y_true):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) globally
    across the entire dataset to avoid batch-averaging bias.

    This function accumulates the Sum of Squared Errors (SSE) and counts
    implicitly by operating on the full concatenated arrays of predictions
    and targets.

    Args:
        y_pred (np.ndarray or torch.Tensor): Predictions. Shape can be
            (N_samples, N_channels) or (N_samples, Seq_Len, N_channels).
        y_true (np.ndarray or torch.Tensor): Ground truth. Shape must match y_pred.

    Returns:
        float: The global MCRMSE score.
        dict: A dictionary containing the RMSE for each scored column.
    """
    # Convert Tensors to Numpy arrays if necessary
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Ensure inputs are float32 for consistent precision
    y_pred = y_pred.astype(np.float32)
    y_true = y_true.astype(np.float32)

    # Check for shape mismatch
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"
        )

    # Flatten sequence dimension if present (e.g., (Batch, Seq, Channels) -> (Batch*Seq, Channels))
    if y_pred.ndim == 3:
        num_channels = y_pred.shape[-1]
        y_pred = y_pred.reshape(-1, num_channels)
        y_true = y_true.reshape(-1, num_channels)

    # Select scored columns
    # The dataset provides 5 columns, but only 3 are scored: reactivity, deg_Mg_pH10, deg_Mg_50C.
    # Indices are defined in Config.SCORED_TARGET_INDICES (usually [0, 1, 3]).
    if y_pred.shape[1] == len(Config.ALL_TARGET_COLS):
        y_pred = y_pred[:, Config.SCORED_TARGET_INDICES]
        y_true = y_true[:, Config.SCORED_TARGET_INDICES]
        col_names = Config.SCORED_TARGET_COLS
    elif y_pred.shape[1] == len(Config.SCORED_TARGET_COLS):
        # Assume already filtered
        col_names = Config.SCORED_TARGET_COLS
    else:
        # Fallback for unexpected shapes, though this shouldn't happen with correct usage
        col_names = [f"col_{i}" for i in range(y_pred.shape[1])]

    # Compute MSE per column (Mean Squared Error)
    # axis=0 aggregates over all samples (global calculation)
    mse_per_col = np.mean((y_true - y_pred) ** 2, axis=0)

    # Compute RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Compute MCRMSE (Mean of column-wise RMSEs)
    mcrmse = np.mean(rmse_per_col)

    # Create breakdown dictionary
    metrics_dict = {name: rmse for name, rmse in zip(col_names, rmse_per_col)}
    metrics_dict["MCRMSE"] = mcrmse

    return mcrmse, metrics_dict
