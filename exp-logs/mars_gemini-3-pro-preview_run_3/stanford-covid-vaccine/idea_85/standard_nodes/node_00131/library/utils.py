import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed=42):
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


def get_device():
    """
    Selects the appropriate device for computation.

    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mcrmse_loss(pred, target):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss
    across all 5 target columns for training.

    Automatically slices the prediction tensor to match the target length
    (e.g., slicing 107 positions to 68) if dimensions differ.

    Args:
        pred (torch.Tensor): Predicted values of shape (Batch, Seq_Len_Pred, 5).
        target (torch.Tensor): Ground truth values of shape (Batch, Seq_Len_Target, 5).

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    # Slice prediction to match target length (usually 68)
    if pred.shape[1] > target.shape[1]:
        pred = pred[:, : target.shape[1], :]

    # Calculate MSE
    mse = (pred - target) ** 2

    # Calculate RMSE per column (averaging over batch and sequence dimensions)
    rmse_per_col = torch.sqrt(torch.mean(mse, dim=(0, 1)))

    # Average RMSEs across the 5 columns
    loss = torch.mean(rmse_per_col)

    return loss


def metric_mcrmse(pred, target):
    """
    Calculates the validation MCRMSE metric strictly according to competition rules.

    1. Slices predictions to the first `Config.PRED_LEN` (68) positions.
    2. Filters for the specific scored columns defined in `Config.SCORED_COLS`:
       ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"].

    Args:
        pred (torch.Tensor): Predicted values of shape (Batch, Seq_Len, 5).
        target (torch.Tensor): Ground truth values of shape (Batch, Scored_Len, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Slice predictions to the scored length (68)
    scored_len = Config.PRED_LEN
    if pred.shape[1] > scored_len:
        pred = pred[:, :scored_len, :]

    # Identify indices of the scored columns
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Expected indices: [0, 1, 3]

    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS

    col_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    # Filter tensors to only include scored columns
    pred_scored = pred[:, :, col_indices]
    target_scored = target[:, :, col_indices]

    # Calculate RMSE per column
    mse = (pred_scored - target_scored) ** 2
    rmse_per_col = torch.sqrt(torch.mean(mse, dim=(0, 1)))

    # Average RMSEs across the scored columns
    score = torch.mean(rmse_per_col)

    return score.item()
