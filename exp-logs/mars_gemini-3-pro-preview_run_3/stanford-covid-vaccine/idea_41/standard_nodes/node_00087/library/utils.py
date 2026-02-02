import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_true, y_pred):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Logic:
    1. Slices predictions and targets to the scored sequence length (first 68 bases).
    2. Filters for the specific scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    3. Computes RMSE for each column independently.
    4. Returns the mean of these RMSE values.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (Batch, Seq_Len, 5).
        y_pred (torch.Tensor): Predicted tensor of shape (Batch, Seq_Len, 5).

    Returns:
        torch.Tensor: Scalar MCRMSE loss.
    """
    # Identify indices of columns that contribute to the score
    # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Indices should be [0, 1, 3]
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # 1. Slice to the scored sequence length (Config.PRED_LEN = 68)
    # We assume the sequence dimension is dim=1
    y_true_sliced = y_true[:, : Config.PRED_LEN, :]
    y_pred_sliced = y_pred[:, : Config.PRED_LEN, :]

    # 2. Filter for the scored columns
    # We assume the channel/target dimension is dim=2
    y_true_filtered = y_true_sliced[:, :, scored_indices]
    y_pred_filtered = y_pred_sliced[:, :, scored_indices]

    # 3. Calculate MSE per column
    # Average over Batch (dim 0) and Sequence (dim 1) dimensions
    mse = torch.mean((y_true_filtered - y_pred_filtered) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # 4. Average the RMSEs across the scored columns
    score = torch.mean(rmse)

    return score
