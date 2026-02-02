import os
import random
import numpy as np
import torch
from library import config


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

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


def mcrmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    According to the task description, this loss is calculated over the
    entire sequence length (0-107) during training to anchor the model
    predictions in the tail region.

    Args:
        pred (torch.Tensor): Predicted values of shape (Batch, Seq_Len, Targets).
        target (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Targets).

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    # Calculate Mean Squared Error per column (averaging over Batch and Sequence)
    # shape: (Targets,)
    col_mse = torch.mean((pred - target) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    col_rmse = torch.sqrt(col_mse)

    # Average across columns to get MCRMSE
    loss = torch.mean(col_rmse)

    return loss


def calculate_global_mcrmse(all_preds: np.ndarray, all_targets: np.ndarray) -> float:
    """
    Calculates the global MCRMSE metric for validation.

    This metric is computed:
    1. Only on the scored sequence positions (0 to SEQ_SCORED).
    2. Only on the scored target columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    3. By aggregating squared errors globally before rooting (Global RMSE).

    Args:
        all_preds (np.ndarray): Array of predictions (N_samples, Seq_Len, 5).
        all_targets (np.ndarray): Array of targets (N_samples, Seq_Len, 5).

    Returns:
        float: The global MCRMSE score.
    """
    # Select only the scored positions (first 68 bases)
    # Shape becomes: (N_samples, 68, 5)
    preds_scored = all_preds[:, : config.SEQ_SCORED, :]
    targets_scored = all_targets[:, : config.SEQ_SCORED, :]

    # Select only the scored columns
    # config.SCORED_TARGET_INDICES is [0, 1, 3]
    # Shape becomes: (N_samples, 68, 3)
    preds_filtered = preds_scored[:, :, config.SCORED_TARGET_INDICES]
    targets_filtered = targets_scored[:, :, config.SCORED_TARGET_INDICES]

    # Calculate Squared Error
    squared_diff = (preds_filtered - targets_filtered) ** 2

    # Calculate Mean Squared Error per column globally (over samples and sequence positions)
    # axis=(0, 1) aggregates over samples and sequence length
    mse_per_col = np.mean(squared_diff, axis=(0, 1))

    # Calculate RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate Mean of RMSEs
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
