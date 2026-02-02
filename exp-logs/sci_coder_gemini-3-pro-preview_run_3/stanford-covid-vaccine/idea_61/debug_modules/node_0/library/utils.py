import os
import random
import numpy as np
import torch
import torch.nn as nn
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
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    Calculates the RMSE for each target column separately and then averages them.
    MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred (torch.Tensor): Predicted values. Shape (N, C) or (N, L, C).
            y_true (torch.Tensor): Ground truth values. Shape (N, C) or (N, L, C).

        Returns:
            torch.Tensor: Scalar MCRMSE loss.
        """
        # Ensure inputs are floating point
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Calculate Squared Error
        squared_diff = (y_pred - y_true) ** 2

        # Determine dimensions to average over.
        # We want to average over all dimensions except the last one (Channels/Targets).
        # If input is (Batch, Seq, Channels), dim=3, we reduce over (0, 1).
        # If input is (Batch, Channels), dim=2, we reduce over (0).
        dims_to_reduce = list(range(y_pred.dim() - 1))

        # Mean Squared Error per column
        mse = torch.mean(squared_diff, dim=dims_to_reduce)

        # Root Mean Squared Error per column
        rmse = torch.sqrt(mse)

        # Mean of RMSEs (MCRMSE)
        mcrmse = torch.mean(rmse)

        return mcrmse


def mcrmse_loss(y_pred, y_true):
    """
    Functional wrapper for MCRMSE calculation.
    Handles both PyTorch tensors and NumPy arrays.

    Args:
        y_pred (Union[torch.Tensor, np.ndarray]): Predicted values.
        y_true (Union[torch.Tensor, np.ndarray]): Ground truth values.

    Returns:
        Union[torch.Tensor, float]: MCRMSE loss. Returns float if inputs were NumPy arrays.
    """
    is_numpy = False

    # Handle NumPy inputs
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)
        is_numpy = True
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
        is_numpy = True

    # Calculate Loss
    criterion = MCRMSELoss()
    loss = criterion(y_pred, y_true)

    if is_numpy:
        return loss.item()

    return loss
