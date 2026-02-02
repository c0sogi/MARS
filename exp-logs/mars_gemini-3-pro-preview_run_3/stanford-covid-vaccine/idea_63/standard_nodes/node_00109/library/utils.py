import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def seed_everything(seed: int = 42):
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


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.

    Calculates the RMSE for each column (target variable) separately and then
    takes the average across columns. This is used as the training objective
    over all 5 target columns.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predictions tensor of shape (Batch, Seq, Columns)
            targets: Ground truth tensor of shape (Batch, Seq, Columns)

        Returns:
            Scalar tensor representing the MCRMSE loss.
        """
        # Calculate MSE per element
        mse = (preds - targets) ** 2

        # Average over Batch and Sequence dimensions to get MSE per column
        # shape: (Columns,)
        mse_per_column = torch.mean(mse, dim=(0, 1))

        # Take Square Root to get RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Average over the columns to get MCRMSE
        mcrmse = torch.mean(rmse_per_column)

        return mcrmse


def calculate_metric(preds, targets):
    """
    Calculates the competition metric: MCRMSE on specific scored columns and positions.

    Logic:
    1. Slices data to the first `Config.SEQ_SCORED` positions.
    2. Filters columns to only `Config.SCORED_TARGETS`.
    3. Computes MCRMSE.

    Args:
        preds: Predictions (numpy array or torch tensor).
        targets: Ground truth (numpy array or torch tensor).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert to torch tensors if inputs are numpy arrays
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Ensure inputs are on CPU for metric calculation
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    # 1. Slice to scored sequence length
    # The model might output 107 positions, but we only score the first 68.
    seq_scored = Config.SEQ_SCORED
    preds_sliced = preds[:, :seq_scored, :]
    targets_sliced = targets[:, :seq_scored, :]

    # 2. Identify indices of scored columns
    # Config.TARGET_COLS contains all 5 columns in order.
    # Config.SCORED_TARGETS contains the subset [reactivity, deg_Mg_pH10, deg_Mg_50C].
    all_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_TARGETS

    scored_indices = [i for i, col in enumerate(all_cols) if col in scored_cols]

    # Select only the scored columns
    preds_filtered = preds_sliced[:, :, scored_indices]
    targets_filtered = targets_sliced[:, :, scored_indices]

    # 3. Compute MCRMSE
    mse = (preds_filtered - targets_filtered) ** 2

    # Mean over batch (0) and sequence (1) dimensions
    mse_per_col = torch.mean(mse, dim=(0, 1))

    # RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col)

    # Mean of RMSEs
    metric = torch.mean(rmse_per_col)

    return metric.item()
