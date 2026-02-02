import os
import random
import ast
import numpy as np
import torch
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_list_column(x):
    """
    Parses a stringified list from a CSV column into a numpy array of float32.
    Returns an empty array if parsing fails.
    """
    try:
        # Check if x is a valid string representation of a list
        if isinstance(x, str):
            return np.array(ast.literal_eval(x), dtype=np.float32)
        return np.array([], dtype=np.float32)
    except (ValueError, SyntaxError):
        return np.array([], dtype=np.float32)


def mcrmse_loss(y_pred, y_true):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    Args:
        y_pred (torch.Tensor): Predicted values of shape (Batch, Seq_Len, 5).
        y_true (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, 5).

    Returns:
        torch.Tensor: Scalar MCRMSE loss.
    """
    # Define the indices of the columns that are scored
    # 0: reactivity
    # 1: deg_Mg_pH10
    # 3: deg_Mg_50C
    # (deg_pH10 at index 2 and deg_50C at index 4 are not scored)
    scored_indices = [0, 1, 3]

    # Ensure we only look at the scored sequence positions (0 to 67)
    # y_pred might be length 107, but y_true is usually length 68 for the scored part
    seq_scored = Config.PRED_LEN

    # Slice to the scored length
    y_pred_scored = y_pred[:, :seq_scored, :]
    y_true_scored = y_true[:, :seq_scored, :]

    # Select only the scored columns
    y_pred_filtered = y_pred_scored[:, :, scored_indices]
    y_true_filtered = y_true_scored[:, :, scored_indices]

    # Compute MSE for each element
    mse = (y_pred_filtered - y_true_filtered) ** 2

    # Compute RMSE for each column:
    # 1. Average MSE over batch (dim 0) and sequence (dim 1)
    # 2. Take sqrt to get RMSE per column
    mse_per_col = torch.mean(mse, dim=(0, 1))
    rmse_per_col = torch.sqrt(mse_per_col)

    # MCRMSE is the mean of the column-wise RMSEs
    loss = torch.mean(rmse_per_col)

    return loss
