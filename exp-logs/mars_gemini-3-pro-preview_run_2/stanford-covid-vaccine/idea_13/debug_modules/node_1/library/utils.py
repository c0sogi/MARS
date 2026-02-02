import os
import ast
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, Numpy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_list_column(x):
    """
    Parses a stringified list from a CSV column into a numpy array of floats.

    Args:
        x (str): A string representation of a list (e.g., "[0.1, 0.2, 0.3]").

    Returns:
        np.ndarray: A numpy array of type float32. Returns an empty array if parsing fails.
    """
    try:
        # ast.literal_eval safely evaluates a string containing a Python literal
        if not isinstance(x, str):
            return np.array([], dtype=np.float32)
        val = ast.literal_eval(x)
        return np.array(val, dtype=np.float32)
    except (ValueError, SyntaxError, TypeError):
        return np.array([], dtype=np.float32)


def calculate_global_mcrmse(preds, targets):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) globally.

    This function computes the RMSE for each column across the entire dataset
    (accumulating SSE and counts) before averaging the column RMSEs. This avoids
    the statistical bias introduced by averaging RMSEs calculated per-batch.

    Args:
        preds (np.ndarray): Predictions array. shape (N_samples, Seq_Len, N_cols)
                            or (N_total_rows, N_cols).
        targets (np.ndarray): Ground truth array. Must match shape of preds.

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    preds = np.asarray(preds)
    targets = np.asarray(targets)

    # Check if shapes match
    if preds.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch in metric calculation: preds {preds.shape} vs targets {targets.shape}"
        )

    # Flatten to 2D: (Total_Observations, N_Columns)
    # If input is (N_samples, Seq_Len, N_cols), this flattens samples and sequence positions
    # to treat every position as an independent observation for the column-wise metric.
    if preds.ndim == 3:
        preds_flat = preds.reshape(-1, preds.shape[-1])
        targets_flat = targets.reshape(-1, targets.shape[-1])
    else:
        preds_flat = preds
        targets_flat = targets

    # Calculate Squared Error for all elements
    squared_error = (preds_flat - targets_flat) ** 2

    # Calculate Mean Squared Error for each column (averaging over all rows/samples)
    mse_per_column = np.mean(squared_error, axis=0)

    # Calculate RMSE for each column
    rmse_per_column = np.sqrt(mse_per_column)

    # Calculate Mean of RMSEs (MCRMSE) across the columns
    mcrmse = np.mean(rmse_per_column)

    return mcrmse
