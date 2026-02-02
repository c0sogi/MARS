import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def calculate_mcrmse(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated by:
    1. Computing the RMSE for each target column (reactivity, deg_Mg_pH10, deg_Mg_50C) independently.
    2. Taking the arithmetic mean of these column-wise RMSE values.

    This aligns with the competition metric and the 'Metric Correction' strategy.

    Args:
        y_true (np.ndarray or torch.Tensor): Ground truth values.
            Expected shape: (Batch, Seq_Len, Num_Targets) or (N, Num_Targets).
        y_pred (np.ndarray or torch.Tensor): Predicted values.
            Expected shape: (Batch, Seq_Len, Num_Targets) or (N, Num_Targets).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Ensure inputs are float32 for consistent precision
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    # Flatten the arrays if they are 3D (Batch, Seq, Targets) -> (Batch*Seq, Targets)
    # We assume the last dimension represents the different target columns.
    if y_true.ndim == 3:
        num_targets = y_true.shape[-1]
        y_true = y_true.reshape(-1, num_targets)
        y_pred = y_pred.reshape(-1, num_targets)

    # 1. Calculate Mean Squared Error (MSE) for each column
    # axis=0 aggregates over the samples, leaving one value per target column
    mse_per_column = np.mean((y_true - y_pred) ** 2, axis=0)

    # 2. Calculate Root Mean Squared Error (RMSE) for each column
    rmse_per_column = np.sqrt(mse_per_column)

    # 3. Calculate the Mean of the column-wise RMSEs
    mcrmse = np.mean(rmse_per_column)

    return float(mcrmse)
