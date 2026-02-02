import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility
    across PyTorch, NumPy, and Python's random module.

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


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.
    Calculates the RMSE for each column independently and then takes the mean
    across columns. This is the primary loss function for the task.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions of shape (Batch, Seq, Channels) or (N, Channels).
            target (torch.Tensor): Ground truth of shape (Batch, Seq, Channels) or (N, Channels).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Flatten to (N, Channels) to handle arbitrary batch/sequence dimensions
        pred_flat = pred.view(-1, pred.size(-1))
        target_flat = target.view(-1, target.size(-1))

        # Calculate Mean Squared Error per column
        mse = torch.mean((pred_flat - target_flat) ** 2, dim=0)

        # Calculate Root Mean Squared Error per column
        rmse = torch.sqrt(mse)

        # Return the mean of the column-wise RMSEs
        return torch.mean(rmse)


def calculate_metric(pred, target):
    """
    Calculates the validation MCRMSE metric specifically for the scored positions and columns.

    Logic:
    1. Slices predictions to the first `seq_scored` positions (68).
    2. Filters for the 3 scored columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3).
    3. Computes MCRMSE on this subset.

    Args:
        pred (torch.Tensor): Predictions of shape (B, Seq_Len, 5).
                             Can be full length (107) or already sliced (68).
        target (torch.Tensor): Ground truth of shape (B, Seq_Scored, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    seq_scored = Config.SEQ_SCORED

    # 1. Slice sequence dimension if necessary
    # If pred is (B, 107, 5), slice to (B, 68, 5)
    if pred.size(1) > seq_scored:
        pred = pred[:, :seq_scored, :]

    # 2. Filter for scored columns
    # Columns: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
    # Indices: 0, 1, 3 are scored.
    scored_indices = [0, 1, 3]

    pred_scored = pred[:, :, scored_indices]
    target_scored = target[:, :, scored_indices]

    # 3. Compute MCRMSE
    # Flatten to (N, 3)
    pred_flat = pred_scored.reshape(-1, len(scored_indices))
    target_flat = target_scored.reshape(-1, len(scored_indices))

    mse = torch.mean((pred_flat - target_flat) ** 2, dim=0)
    rmse = torch.sqrt(mse)
    mcrmse = torch.mean(rmse)

    return mcrmse.item()
