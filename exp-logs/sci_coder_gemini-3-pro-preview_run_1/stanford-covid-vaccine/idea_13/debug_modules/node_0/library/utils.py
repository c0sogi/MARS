import os
import random
import numpy as np
import torch
from library.config import Config


def seed_everything(seed: int = 42) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

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


def mcrmse(y_true: torch.Tensor, y_pred: torch.Tensor, num_scored: int = None) -> float:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is calculated as the average of the RMSE values for each target column.
    Only the first `num_scored` positions in the sequence are considered.

    Args:
        y_true (torch.Tensor): Ground truth tensor of shape (Batch, Seq_Len, Num_Targets).
        y_pred (torch.Tensor): Predicted tensor of shape (Batch, Seq_Len, Num_Targets).
        num_scored (int, optional): Number of positions to score from the start.
                                    Defaults to Config.SEQ_SCORED.

    Returns:
        float: The calculated MCRMSE score.
    """
    if num_scored is None:
        num_scored = Config.SEQ_SCORED

    # Ensure inputs are tensors
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true, dtype=torch.float32)
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

    # Ensure tensors are on the same device
    if y_true.device != y_pred.device:
        y_true = y_true.to(y_pred.device)

    # Slice the tensors to consider only the scored positions
    # Assumes shape: [Batch, Seq_Length, Num_Targets]
    # We slice dim 1 (sequence length) up to num_scored
    y_true_scored = y_true[:, :num_scored, :]
    y_pred_scored = y_pred[:, :num_scored, :]

    # Calculate Squared Error: (y - y_hat)^2
    squared_error = (y_true_scored - y_pred_scored) ** 2

    # Calculate Mean Squared Error per target column
    # Average over Batch (dim 0) and Sequence Position (dim 1)
    # Result shape: (Num_Targets,)
    mse_per_target = torch.mean(squared_error, dim=(0, 1))

    # Calculate Root Mean Squared Error per target column
    rmse_per_target = torch.sqrt(mse_per_target)

    # Calculate the Mean of the columnwise RMSEs
    mcrmse_score = torch.mean(rmse_per_target)

    return mcrmse_score.item()
