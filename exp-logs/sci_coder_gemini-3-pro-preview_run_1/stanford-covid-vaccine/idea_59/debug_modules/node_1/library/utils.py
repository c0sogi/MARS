import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_metric(y_true, y_pred, pred_len=Config.PRED_LEN):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric computes the RMSE for each of the 3 scored columns (reactivity,
    deg_Mg_pH10, deg_Mg_50C) on the first `pred_len` positions and returns the
    average of these RMSEs.

    Args:
        y_true: Ground truth values. Shape (Batch, Seq_Len, 3).
                Can be a numpy array or torch Tensor.
        y_pred: Predicted values. Shape (Batch, Seq_Len, 3).
                Can be a numpy array or torch Tensor.
        pred_len (int): The number of positions to include in the score (default: 68).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure data is float32 for consistent calculation
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    # Slice the arrays to the scored length (typically first 68 positions)
    # We assume the input shape is (Batch, Seq_Len, Channels)
    if y_true.shape[1] > pred_len:
        y_true = y_true[:, :pred_len, :]
    if y_pred.shape[1] > pred_len:
        y_pred = y_pred[:, :pred_len, :]

    rmses = []
    # The input is expected to have 3 channels corresponding to the scored targets
    num_scored_cols = y_true.shape[2]

    for i in range(num_scored_cols):
        # Flatten the arrays to compute MSE over all samples and positions for this column
        col_true = y_true[:, :, i].flatten()
        col_pred = y_pred[:, :, i].flatten()

        # Calculate MSE and then RMSE for the column
        mse = np.mean((col_true - col_pred) ** 2)
        rmse = np.sqrt(mse)
        rmses.append(rmse)

    # Return the mean of the column-wise RMSEs
    return np.mean(rmses)
