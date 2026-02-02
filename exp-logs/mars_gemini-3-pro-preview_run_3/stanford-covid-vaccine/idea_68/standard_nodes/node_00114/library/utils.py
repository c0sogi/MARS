import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(pred: torch.Tensor, target: torch.Tensor):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    Args:
        pred (torch.Tensor): Predicted values of shape (B, L, C).
        target (torch.Tensor): Ground truth values of shape (B, L, C).

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    # Calculate MSE per element
    mse = (pred - target) ** 2
    # Average over batch and sequence dimensions (0 and 1) to get MSE per column
    mse_per_col = mse.mean(dim=(0, 1))
    # Take sqrt to get RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)
    # Average over columns to get the final loss
    return rmse_per_col.mean()


def metric_mcrmse_scored(
    preds: torch.Tensor, targets: torch.Tensor, seq_scored: int = 68
):
    """
    Calculates the specific MCRMSE metric for the competition evaluation.

    Logic:
    1. Slices predictions and targets to the first `seq_scored` positions.
    2. Computes RMSE for each column globally.
    3. Selects specific columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3).
    4. Returns the mean of these selected RMSEs.

    Args:
        preds (torch.Tensor): Predictions tensor of shape (N, L, 5).
        targets (torch.Tensor): Targets tensor of shape (N, L, 5).
        seq_scored (int): Number of positions to score (default: 68).

    Returns:
        float: The calculated metric value.
    """
    # Slice to scored length
    preds_sliced = preds[:, :seq_scored, :]
    targets_sliced = targets[:, :seq_scored, :]

    # Calculate MSE per column over the entire dataset
    mse = (preds_sliced - targets_sliced) ** 2

    # RMSE per column (averaging over batch and sequence length)
    rmse_per_col = torch.sqrt(mse.mean(dim=(0, 1)))

    # Select specific columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    # We ignore deg_pH10(2) and deg_50C(4) for the metric
    selected_indices = [0, 1, 3]

    if preds.is_cuda:
        device = preds.device
        indices = torch.tensor(selected_indices, device=device)
        selected_rmse = torch.index_select(rmse_per_col, 0, indices)
    else:
        selected_rmse = rmse_per_col[selected_indices]

    return selected_rmse.mean().item()
