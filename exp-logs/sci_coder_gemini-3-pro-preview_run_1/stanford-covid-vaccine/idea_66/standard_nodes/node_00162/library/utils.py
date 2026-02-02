import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = Config.SEED):
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Defaults to Config.SEED.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mcrmse_loss(
    pred: torch.Tensor, target: torch.Tensor, num_scored: int = Config.PRED_LEN
) -> torch.Tensor:
    """
    Calculates the MCRMSE (Mean Columnwise Root Mean Squared Error) loss.
    This function is differentiable and can be used as a loss function or for monitoring.

    Logic:
    1. Slice inputs to the scored sequence length (first 68 bases).
    2. Calculate MSE for each column (target) across all samples and sequence positions.
    3. Take the square root of the MSE to get RMSE per column.
    4. Average the RMSE values across columns.

    Args:
        pred: Predicted tensor of shape (Batch, Seq_Len, Channels).
        target: Ground truth tensor of shape (Batch, Seq_Len, Channels).
        num_scored: Number of sequence positions to score. Defaults to Config.PRED_LEN (68).

    Returns:
        torch.Tensor: The calculated MCRMSE loss (scalar).
    """
    # Slice to the scored sequence length
    pred_scored = pred[:, :num_scored, :]
    target_scored = target[:, :num_scored, :]

    # Calculate MSE per column across batch and sequence dimensions
    # dim=(0, 1) aggregates over Batch (0) and Sequence (1)
    mse_per_col = torch.mean((pred_scored - target_scored) ** 2, dim=(0, 1))

    # Calculate RMSE per column (adding epsilon for numerical stability)
    rmse_per_col = torch.sqrt(mse_per_col + 1e-6)

    # Calculate the mean of RMSEs across columns
    loss = torch.mean(rmse_per_col)

    return loss


def mcrmse_metric(pred, target, num_scored: int = Config.PRED_LEN) -> float:
    """
    Calculates the MCRMSE metric for evaluation purposes.
    Accepts either torch Tensors or numpy arrays.

    Args:
        pred: Predictions (torch.Tensor or np.ndarray).
        target: Ground truth (torch.Tensor or np.ndarray).
        num_scored: Number of sequence positions to score. Defaults to Config.PRED_LEN (68).

    Returns:
        float: The MCRMSE score.
    """
    # Convert tensors to numpy arrays if necessary
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Slice to the scored sequence length
    pred_scored = pred[:, :num_scored, :]
    target_scored = target[:, :num_scored, :]

    # Calculate MSE per column across batch and sequence dimensions
    # axis=(0, 1) aggregates over Batch and Sequence
    mse_per_col = np.mean((pred_scored - target_scored) ** 2, axis=(0, 1))

    # Calculate RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Calculate the mean of RMSEs across columns
    score = np.mean(rmse_per_col)

    return float(score)
