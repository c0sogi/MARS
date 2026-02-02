import os
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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metric(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the scored columns.

    Logic:
    1. Slices data to the first 68 positions (Config.PRED_LEN).
    2. Filters for the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.
    3. Computes RMSE for each column over all samples and positions.
    4. Returns the mean of these RMSE values.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth values of shape (N, seq_len, 5).
        y_pred (torch.Tensor or np.ndarray): Predicted values of shape (N, seq_len, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # 1. Slice to the scored sequence length
    # The competition only scores the first 68 bases.
    y_true = y_true[:, : Config.PRED_LEN, :]
    y_pred = y_pred[:, : Config.PRED_LEN, :]

    # 2. Filter for scored columns
    # We need to map the column names to indices.
    # Config.TARGET_COLS contains all 5 columns.
    # Config.SCORED_COLS contains the subset of 3 columns to score.
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Select only the scored columns
    # Shape becomes (N, 68, 3)
    y_true_scored = y_true[:, :, scored_indices]
    y_pred_scored = y_pred[:, :, scored_indices]

    # 3. Compute RMSE per column
    # We aggregate over samples (axis 0) and sequence positions (axis 1)
    mse_per_col = np.mean((y_true_scored - y_pred_scored) ** 2, axis=(0, 1))
    rmse_per_col = np.sqrt(mse_per_col)

    # 4. Compute Mean of RMSEs (MCRMSE)
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
