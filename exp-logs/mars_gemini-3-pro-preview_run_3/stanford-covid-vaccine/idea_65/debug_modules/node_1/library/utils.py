import os
import json
import hashlib
import torch
import numpy as np
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Delegates to the Config class method to avoid code duplication.
    """
    Config.set_seed(seed)


def get_md5_hash(obj):
    """
    Generates an MD5 hash for a Python object (e.g., dictionary configuration).
    This is used to create unique identifiers for caching processed data.

    Args:
        obj: The Python object (dict, list, etc.) to hash.

    Returns:
        str: The hexadecimal MD5 hash string.
    """
    dhash = hashlib.md5()
    # Sort keys to ensure consistent hashing for dictionaries regardless of key order
    encoded = json.dumps(obj, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def MCRMSE(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function performs the following steps required by the competition metric:
    1. Slices the input tensors to the first 68 positions (Config.PRED_LEN).
    2. Filters the tensors to include only the scored columns (Config.SCORED_INDICES).
    3. Calculates the RMSE for each column individually.
    4. Returns the average of these RMSE values.

    Args:
        y_true: Ground truth tensor or array. Expected shape (Batch, Seq_Len, 5).
        y_pred: Predicted tensor or array. Expected shape (Batch, Seq_Len, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Ensure inputs are torch tensors
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Move tensors to CPU for calculation and detach from graph
    y_true = y_true.detach().cpu()
    y_pred = y_pred.detach().cpu()

    # 1. Slice to the scored sequence length (first 68 positions)
    # If the input length is greater than PRED_LEN (e.g., 107), we slice it.
    if y_true.shape[1] > Config.PRED_LEN:
        y_true = y_true[:, : Config.PRED_LEN, :]
    if y_pred.shape[1] > Config.PRED_LEN:
        y_pred = y_pred[:, : Config.PRED_LEN, :]

    # 2. Filter for the scored columns only
    # Config.SCORED_INDICES = [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C)
    y_true_scored = y_true[:, :, Config.SCORED_INDICES]
    y_pred_scored = y_pred[:, :, Config.SCORED_INDICES]

    # 3. Calculate RMSE per column
    # Calculate Squared Error
    squared_error = (y_true_scored - y_pred_scored) ** 2

    # Mean Squared Error (MSE) per column: Average over Batch (dim 0) and Sequence (dim 1)
    mse = torch.mean(squared_error, dim=(0, 1))

    # Root Mean Squared Error (RMSE) per column
    rmse = torch.sqrt(mse)

    # 4. Average the RMSE values across the columns
    mcrmse = torch.mean(rmse)

    return mcrmse.item()
