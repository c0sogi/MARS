import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated by:
    1. Computing the RMSE for each of the target columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    2. Taking the average of these column-wise RMSEs.

    This strictly follows the evaluation protocol of averaging the RMSE of each column
    rather than averaging the RMSE of each position.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
                                             Expected shape: (Batch, Seq_Len, Num_Targets)
                                             or (Total_Samples, Num_Targets).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
                                             Expected shape: same as y_true.

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are numpy arrays with float64 precision for metric calculation
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Check shapes
    assert (
        y_true.shape == y_pred.shape
    ), f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"

    # Calculate MSE per column
    # If 3D array (Batch, Seq, Channels), average over axes 0 and 1
    if y_true.ndim == 3:
        mse_per_col = np.mean((y_true - y_pred) ** 2, axis=(0, 1))
    # If 2D array (Samples, Channels), average over axis 0
    elif y_true.ndim == 2:
        mse_per_col = np.mean((y_true - y_pred) ** 2, axis=0)
    else:
        raise ValueError(f"Unexpected input dimension: {y_true.ndim}. Expected 2 or 3.")

    # Calculate RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate MCRMSE (average of column RMSEs)
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)


def print_metric(name, value):
    """
    Prints a metric with full precision as required.

    Args:
        name (str): Name of the metric.
        value (float): Value of the metric.
    """
    print(f"{name}: {value}")
