import torch
import numpy as np
import random
import os
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across standard libraries.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(pred, target):
    """
    Calculates the MCRMSE loss for training on all 5 targets.

    Formula: Mean of Column-wise RMSEs.

    Args:
        pred (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5) or (Batch, Pred_Len, 5).
        target (torch.Tensor): Ground truth of shape (Batch, Pred_Len, 5).

    Returns:
        torch.Tensor: Scalar loss value with gradient attached.
    """
    # Slice predictions to match target length (seq_scored=68)
    # Target is typically (Batch, 68, 5)
    if pred.shape[1] > target.shape[1]:
        pred = pred[:, : target.shape[1], :]

    # Calculate MSE for each column (averaging over batch and sequence length)
    # shape: (5,)
    mse = torch.mean((pred - target) ** 2, dim=(0, 1))

    # Calculate RMSE for each column
    rmse = torch.sqrt(mse)

    # Return the mean of RMSEs across the 5 columns
    return torch.mean(rmse)


def competition_metric(pred, target):
    """
    Calculates the competition metric (MCRMSE) on the 3 scored columns.

    Scored Columns: reactivity, deg_Mg_pH10, deg_Mg_50C (Indices 0, 1, 3).

    Args:
        pred (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5) or (Batch, Pred_Len, 5).
        target (torch.Tensor): Ground truth of shape (Batch, Pred_Len, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Slice predictions to match target length (seq_scored=68)
    if pred.shape[1] > target.shape[1]:
        pred = pred[:, : target.shape[1], :]

    # Filter for the scored columns only
    pred_scored = pred[:, :, Config.SCORED_INDICES]
    target_scored = target[:, :, Config.SCORED_INDICES]

    # Calculate MSE for each scored column
    mse = torch.mean((pred_scored - target_scored) ** 2, dim=(0, 1))

    # Calculate RMSE
    rmse = torch.sqrt(mse)

    # Return mean of RMSEs
    return torch.mean(rmse).item()
