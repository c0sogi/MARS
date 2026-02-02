import os
import random
import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

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


class MCRMSE(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    Calculates the mean of the RMSE values for each target column.
    Formula: (1/Nt) * Sum_j( sqrt( (1/n) * Sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Where:
        Nt = Number of target columns
        n  = Total number of predictions (samples * sequence_positions)
        j  = Index of target column
        i  = Index of prediction instance
    """

    def __init__(self):
        super(MCRMSE, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Forward pass for MCRMSE.

        Args:
            y_pred (torch.Tensor): Predicted values. Shape (Batch, ..., Columns).
            y_true (torch.Tensor): Ground truth values. Shape (Batch, ..., Columns).

        Returns:
            torch.Tensor: Scalar MCRMSE score.
        """
        # Ensure inputs are tensors
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred, dtype=torch.float32)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32)

        # Get the number of columns (targets)
        # We assume the last dimension represents the different target variables
        num_columns = y_pred.shape[-1]

        # Flatten all dimensions except the last one (columns)
        # This aggregates samples and sequence positions together into a single 'n' dimension
        y_pred_flat = y_pred.view(-1, num_columns)
        y_true_flat = y_true.view(-1, num_columns)

        # Calculate MSE for each column independently
        mse = torch.mean((y_true_flat - y_pred_flat) ** 2, dim=0)

        # Calculate RMSE for each column
        rmse = torch.sqrt(mse)

        # Calculate Mean of RMSEs across columns
        mcrmse_val = torch.mean(rmse)

        return mcrmse_val


def mcrmse(y_true, y_pred):
    """
    Functional wrapper for MCRMSE metric.

    Args:
        y_true (array-like): Ground truth values. Can be numpy array or torch tensor.
        y_pred (array-like): Predicted values. Can be numpy array or torch tensor.

    Returns:
        torch.Tensor: MCRMSE score.
    """
    criterion = MCRMSE()

    # Convert numpy arrays to tensors if necessary
    if isinstance(y_true, np.ndarray):
        y_true = torch.from_numpy(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.from_numpy(y_pred)

    return criterion(y_pred, y_true)
