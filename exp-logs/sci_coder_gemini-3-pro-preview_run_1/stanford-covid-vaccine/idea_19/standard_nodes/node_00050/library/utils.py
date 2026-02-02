import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    Enforces deterministic CuDNN behavior.

    Args:
        seed (int): The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Ensure deterministic behavior to guarantee reproducible results
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the arithmetic mean of the RMSE values for each
    target column. This aligns with the competition metric where N_t is the
    number of scored columns.

    Expected inputs should correspond to the 3 scored columns:
    1. reactivity
    2. deg_Mg_pH10
    3. deg_Mg_50C

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values. Shape (N, 3).
        y_pred (np.ndarray or torch.Tensor): Predicted values. Shape (N, 3).

    Returns:
        float: The MCRMSE score.
    """
    # Detach and convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are float64 numpy arrays for precision
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # 1. Calculate Mean Squared Error for each column (axis 0 is samples)
    # Result shape: (num_columns,)
    column_mse = np.mean(np.square(y_true - y_pred), axis=0)

    # 2. Calculate Root Mean Squared Error for each column
    column_rmse = np.sqrt(column_mse)

    # 3. Calculate the Mean of the column RMSEs
    score = np.mean(column_rmse)

    return float(score)
