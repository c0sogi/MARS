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
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior for CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mcrmse_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    The metric is defined as the average of the RMSE values calculated for each
    target column independently.

    Args:
        y_pred (torch.Tensor): Predicted values. Shape (..., num_targets).
        y_true (torch.Tensor): Ground truth values. Shape (..., num_targets).

    Returns:
        torch.Tensor: Scalar MCRMSE loss.
    """
    # Ensure inputs are tensors
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true)

    # Calculate Squared Error
    squared_error = (y_true - y_pred) ** 2

    # Calculate Mean Squared Error per column.
    # We average over all dimensions except the last one (which are the target columns).
    # For example, if input is (Batch, Seq_Len, 5), we average over Batch and Seq_Len.
    dims_to_reduce = list(range(y_true.dim() - 1))
    mse = torch.mean(squared_error, dim=dims_to_reduce)

    # Calculate RMSE per column
    rmse = torch.sqrt(mse)

    # Calculate Mean of RMSEs across columns (the MCRMSE)
    mcrmse = torch.mean(rmse)

    return mcrmse
