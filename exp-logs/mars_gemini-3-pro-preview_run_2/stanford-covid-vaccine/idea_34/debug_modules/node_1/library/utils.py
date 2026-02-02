import os
import ast
import random
import numpy as np
import torch
from library.config import SCORED_INDICES


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
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_list_column(x):
    """
    Parses a string representation of a list (e.g., from a CSV file) into a numpy array.

    Args:
        x (str): String representation of a list, e.g., "[0.1, 0.2, 0.3]".

    Returns:
        np.ndarray: A numpy array of floats. Returns an empty array if parsing fails.
    """
    try:
        # ast.literal_eval is safer than eval
        return np.array(ast.literal_eval(x), dtype=np.float32)
    except (ValueError, SyntaxError):
        return np.array([], dtype=np.float32)


def mcrmse_loss(pred, target, scored_indices=SCORED_INDICES):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    The metric is defined as the average of the RMSE values for each scored column.
    Only the columns specified by scored_indices are used in the calculation.

    Args:
        pred (torch.Tensor): Predicted values of shape (Batch, ..., Channels).
        target (torch.Tensor): Ground truth values of shape (Batch, ..., Channels).
        scored_indices (list, optional): List of indices corresponding to the columns
                                         to be scored. Defaults to SCORED_INDICES from config.
                                         (reactivity, deg_Mg_pH10, deg_Mg_50C).

    Returns:
        torch.Tensor: Scalar tensor containing the MCRMSE loss.
    """
    # Select only the columns that contribute to the score
    # Assuming the last dimension is the channel dimension
    pred_scored = pred[..., scored_indices]
    target_scored = target[..., scored_indices]

    # Calculate MSE for each column
    # We average over all dimensions except the last (channel) dimension
    mse = torch.mean(
        (pred_scored - target_scored) ** 2, dim=list(range(pred_scored.ndim - 1))
    )

    # Calculate RMSE for each column
    rmse = torch.sqrt(mse)

    # Calculate the mean of the RMSEs across the scored columns
    mcrmse = torch.mean(rmse)

    return mcrmse
