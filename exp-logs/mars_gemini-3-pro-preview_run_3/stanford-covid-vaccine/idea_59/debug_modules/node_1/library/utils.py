import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This function is used for the training objective. It computes the loss over
    all 5 target columns as specified in the strategy (Multi-Task Learning).

    Args:
        pred: Predicted tensor of shape (Batch, Seq_Len, 5).
        target: Ground truth tensor of shape (Batch, Scored_Len, 5).

    Returns:
        A scalar Tensor representing the mean of the RMSEs across the 5 columns.
    """
    # Slice predictions to match the scored length (first 68 positions)
    # pred shape: (B, 107, 5) -> (B, 68, 5)
    pred_sliced = pred[:, : Config.PRED_LEN, :]

    # Calculate MSE per element
    mse = (pred_sliced - target) ** 2

    # Average MSE over Batch and Sequence dimensions (dim 0 and 1) to get MSE per column
    # Shape: (5,)
    mse_per_column = torch.mean(mse, dim=(0, 1))

    # Take sqrt to get RMSE per column
    rmse_per_column = torch.sqrt(mse_per_column)

    # Average over the columns to get MCRMSE
    loss = torch.mean(rmse_per_column)

    return loss


def mcrmse_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculates the MCRMSE metric for validation/evaluation.

    This function computes the metric specifically on the 3 scored columns:
    reactivity, deg_Mg_pH10, and deg_Mg_50C, over the first 68 positions.

    Args:
        pred: Predicted tensor of shape (Batch, Seq_Len, 5).
        target: Ground truth tensor of shape (Batch, Scored_Len, 5).

    Returns:
        A float representing the MCRMSE on the scored columns.
    """
    with torch.no_grad():
        # Slice predictions to match the scored length (first 68 positions)
        pred_sliced = pred[:, : Config.PRED_LEN, :]

        # Select only the scored columns [0, 1, 3]
        # 0: reactivity, 1: deg_Mg_pH10, 3: deg_Mg_50C
        pred_scored = pred_sliced[:, :, Config.SCORED_TARGET_INDICES]
        target_scored = target[:, :, Config.SCORED_TARGET_INDICES]

        # Calculate MSE per element
        mse = (pred_scored - target_scored) ** 2

        # Average MSE over Batch and Sequence dimensions
        mse_per_column = torch.mean(mse, dim=(0, 1))

        # Take sqrt to get RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Average over the 3 scored columns
        metric = torch.mean(rmse_per_column)

        return metric.item()
