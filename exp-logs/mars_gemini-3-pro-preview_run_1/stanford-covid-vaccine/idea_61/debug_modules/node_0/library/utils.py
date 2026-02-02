import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the average of the RMSE for each of the
    scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values. Shape (N, 3).
        y_pred (np.ndarray or torch.Tensor): Predicted values. Shape (N, 3).

    Returns:
        float: The MCRMSE score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Calculate Squared Error: (y - y_hat)^2
    squared_error = (y_true - y_pred) ** 2

    # Calculate Mean Squared Error per column (average over samples/rows)
    mse_per_column = np.mean(squared_error, axis=0)

    # Calculate Root Mean Squared Error per column
    rmse_per_column = np.sqrt(mse_per_column)

    # Calculate the average of the column RMSEs
    score = np.mean(rmse_per_column)

    return float(score)
