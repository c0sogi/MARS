import os
import random
import numpy as np
import torch
from library.config import Config


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def mcrmse_loss(pred, target):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for training.
    Optimizes all 5 target columns as per the strategy.

    Args:
        pred (torch.Tensor): Predicted values of shape (Batch, Seq_Len_Pred, 5).
        target (torch.Tensor): Ground truth values of shape (Batch, Seq_Len_Target, 5).

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    # Slice predictions to match the length of the targets (usually 68)
    # pred shape: (B, 107, 5), target shape: (B, 68, 5)
    seq_len_target = target.shape[1]
    pred_sliced = pred[:, :seq_len_target, :]

    # Calculate MSE for each column (averaging over batch and sequence dimensions)
    # dim=(0, 1) aggregates over Batch and Sequence
    mse = torch.mean((pred_sliced - target) ** 2, dim=(0, 1))

    # Calculate RMSE for each column
    rmse = torch.sqrt(mse)

    # Calculate Mean of RMSEs across columns
    loss = torch.mean(rmse)

    return loss


def global_mcrmse_metric(predictions, targets):
    """
    Calculates the MCRMSE metric for validation/evaluation.
    Aggregates all batches first, slices to the scored sequence length,
    and filters for the specific scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).

    Args:
        predictions (list of torch.Tensor): List of prediction tensors (B, 107, 5).
        targets (list of torch.Tensor): List of target tensors (B, 68, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Concatenate all batches
    # We assume inputs are on the same device (likely CPU for validation aggregation)
    preds_cat = torch.cat(predictions, dim=0)
    targets_cat = torch.cat(targets, dim=0)

    # Slice predictions to the scored sequence length (68)
    # This ensures we only evaluate positions for which we have ground truth
    seq_scored = targets_cat.shape[1]
    preds_cat = preds_cat[:, :seq_scored, :]

    # Identify indices of the columns that are actually scored in the competition
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Filter predictions and targets to only include scored columns
    preds_scored = preds_cat[:, :, scored_indices]
    targets_scored = targets_cat[:, :, scored_indices]

    # Calculate MSE per column (averaging over batch and sequence)
    mse = torch.mean((preds_scored - targets_scored) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # Calculate Mean of RMSEs
    mcrmse = torch.mean(rmse)

    return mcrmse.item()
