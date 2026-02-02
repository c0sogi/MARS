import os
import random
import numpy as np
import torch
from library.config import Config


def seed_all(seed=42):
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


def mcrmse_metric(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE)
    for the specific scored columns and sequence positions.

    Logic:
    1. Slices data to the first Config.SEQ_SCORED positions.
    2. Filters columns to Config.SCORED_TARGETS.
    3. Computes RMSE for each column.
    4. Returns the mean of these RMSE values.

    Args:
        y_true (torch.Tensor or np.ndarray): Ground truth of shape (N, seq_len, 5).
        y_pred (torch.Tensor or np.ndarray): Predictions of shape (N, seq_len, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Convert to tensor if numpy
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    # Ensure predictions are on the same device as targets
    if y_true.device != y_pred.device:
        y_pred = y_pred.to(y_true.device)

    # 1. Slice sequence dimension
    # We only score the first SEQ_SCORED positions (e.g., 68).
    y_true_sliced = y_true[:, : Config.SEQ_SCORED, :]
    y_pred_sliced = y_pred[:, : Config.SEQ_SCORED, :]

    # 2. Identify indices of scored columns
    # Config.TARGET_COLS contains all 5 targets
    # Config.SCORED_TARGETS contains the subset used for the leaderboard
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_TARGETS
    ]

    # 3. Filter columns to only the scored ones
    y_true_filtered = y_true_sliced[:, :, scored_indices]
    y_pred_filtered = y_pred_sliced[:, :, scored_indices]

    # 4. Compute MSE per column
    # Calculate mean over batch (dim 0) and sequence (dim 1)
    mse_per_col = torch.mean((y_true_filtered - y_pred_filtered) ** 2, dim=(0, 1))

    # 5. Compute RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)

    # 6. Average RMSE across the scored columns to get MCRMSE
    mcrmse = torch.mean(rmse_per_col)

    return mcrmse.item()
