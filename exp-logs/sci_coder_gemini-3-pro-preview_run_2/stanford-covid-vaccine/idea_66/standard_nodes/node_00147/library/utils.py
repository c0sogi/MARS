import os
import ast
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


def get_device():
    """
    Determines and returns the available PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The device object.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_list_column(x, length=Config.SEQ_LENGTH, pad_value=0.0):
    """
    Parses a string representation of a list (e.g., from a CSV file) into a NumPy array.
    Pads or truncates the array to the specified length.

    Args:
        x (str or list): The input string or list.
        length (int): The desired length of the output array. Defaults to Config.SEQ_LENGTH.
        pad_value (float): The value to use for padding. Defaults to 0.0.

    Returns:
        np.ndarray: A float32 NumPy array of the parsed list, padded to length.
    """
    try:
        if isinstance(x, str):
            # Evaluate the string as a Python literal (list)
            val_list = ast.literal_eval(x)
        elif isinstance(x, (list, tuple, np.ndarray)):
            val_list = x
        else:
            # Fallback for unexpected types (e.g., NaN float)
            val_list = []

        arr = np.array(val_list, dtype=np.float32)

        # Pad if the array is shorter than the target length
        if len(arr) < length:
            padded = np.full(length, pad_value, dtype=np.float32)
            padded[: len(arr)] = arr
            return padded

        # Truncate if longer
        if len(arr) > length:
            return arr[:length]

        return arr

    except Exception:
        # Return a zero-filled array in case of parsing errors
        return np.full(length, pad_value, dtype=np.float32)


def mcrmse(preds, targets, scored_len=Config.SEQ_SCORED):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) using NumPy.
    Only the following columns are scored: reactivity, deg_Mg_pH10, deg_Mg_50C.
    Indices corresponding to these are [0, 1, 3].

    Args:
        preds (np.ndarray): Predicted values. Shape (N, 5, L) or (N, L, 5).
        targets (np.ndarray): Ground truth values. Shape (N, 5, L) or (N, L, 5).
        scored_len (int): The number of positions to score from the start of the sequence.
                          Defaults to Config.SEQ_SCORED.

    Returns:
        float: The calculated MCRMSE score.
    """
    # Indices for: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    scored_cols_indices = [0, 1, 3]

    # Standardize shape to (N, 5, L) (Batch, Channels, Length)
    # If input is (N, L, 5), transpose it to (N, 5, L)
    if preds.ndim == 3 and preds.shape[2] == 5:
        preds = np.transpose(preds, (0, 2, 1))
    if targets.ndim == 3 and targets.shape[2] == 5:
        targets = np.transpose(targets, (0, 2, 1))

    # Extract the relevant columns and sequence positions
    # Shape becomes: (N, 3, scored_len)
    preds_scored = preds[:, scored_cols_indices, :scored_len]
    targets_scored = targets[:, scored_cols_indices, :scored_len]

    # Calculate MSE for each column: Mean over Batch (axis 0) and Length (axis 2)
    # Result shape: (3,)
    mse_per_col = np.mean((preds_scored - targets_scored) ** 2, axis=(0, 2))

    # Calculate RMSE for each column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate the mean of the column RMSEs
    mcrmse_score = np.mean(rmse_per_col)

    return float(mcrmse_score)
