import os
import random
import numpy as np
import torch
from library.config import Config


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_numpy(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the scored targets.

    Logic:
    1. Slices predictions to the first 68 positions (Config.SEQ_SCORED).
    2. Selects only the columns specified in Config.SCORED_TARGETS.
    3. Computes RMSE for each column and returns the mean.

    Args:
        y_true (np.ndarray): Ground truth array of shape (N, 68, 5).
        y_pred (np.ndarray): Predicted array of shape (N, 107, 5) or (N, 68, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Ensure inputs are numpy arrays
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # 1. Slice predictions to match scored sequence length
    # y_pred might be (N, 107, 5), slice to (N, 68, 5)
    if y_pred.shape[1] > Config.SEQ_SCORED:
        y_pred = y_pred[:, : Config.SEQ_SCORED, :]

    # Verify shapes match after slicing
    assert (
        y_true.shape == y_pred.shape
    ), f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"

    # 2. Identify indices of scored columns
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [Config.TARGET_COLS.index(col) for col in Config.SCORED_TARGETS]

    # Filter arrays to keep only scored columns
    # Shape becomes (N, 68, 3)
    y_true_scored = y_true[:, :, scored_indices]
    y_pred_scored = y_pred[:, :, scored_indices]

    # 3. Compute RMSE per column
    # Mean over samples (axis 0) and sequence positions (axis 1)
    # Result is a vector of RMSEs for the 3 columns
    mse_per_col = np.mean((y_true_scored - y_pred_scored) ** 2, axis=(0, 1))
    rmse_per_col = np.sqrt(mse_per_col)

    # 4. Return Mean of RMSEs
    mcrmse = np.mean(rmse_per_col)

    return mcrmse
