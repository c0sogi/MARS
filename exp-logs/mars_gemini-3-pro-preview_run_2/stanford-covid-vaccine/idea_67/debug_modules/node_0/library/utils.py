import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import SCORED_LEN


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        """
        Calculates MCRMSE over the full sequence length (Boundary Anchoring).

        Args:
            inputs: Predictions tensor of shape (Batch, SeqLen, 5).
            targets: Ground truth tensor of shape (Batch, SeqLen, 5).

        Returns:
            Scalar tensor representing the mean columnwise RMSE.
        """
        # Calculate Squared Error per element
        squared_diff = (inputs - targets) ** 2

        # Mean Squared Error per column (averaging over Batch and Sequence dimensions)
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # Root Mean Squared Error per column
        # Adding epsilon for numerical stability
        rmse_per_column = torch.sqrt(mse_per_column + 1e-8)

        # Average RMSE across all 5 columns
        mcrmse = torch.mean(rmse_per_column)

        return mcrmse


def metric_mcrmse(preds, targets):
    """
    Calculates the official MCRMSE metric on the scored subset.

    This function strictly filters the data to:
    1. The first 68 sequence positions (SCORED_LEN).
    2. The 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.

    It computes the global RMSE (accumulating SSE over the entire set)
    rather than averaging batch-wise RMSEs, to avoid optimistic bias.

    Args:
        preds: Numpy array of predictions (N, SeqLen, 5).
        targets: Numpy array of ground truth (N, SeqLen, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # 1. Slice to the scored sequence length (first 68 positions)
    preds_scored = preds[:, :SCORED_LEN, :]
    targets_scored = targets[:, :SCORED_LEN, :]

    # 2. Select the scored columns.
    # Based on standard dataset ordering:
    # 0: reactivity (Scored)
    # 1: deg_Mg_pH10 (Scored)
    # 2: deg_pH10 (Not Scored)
    # 3: deg_Mg_50C (Scored)
    # 4: deg_50C (Not Scored)
    scored_indices = [0, 1, 3]

    preds_selected = preds_scored[:, :, scored_indices]
    targets_selected = targets_scored[:, :, scored_indices]

    # 3. Calculate RMSE per column globally
    # Flatten batch and sequence dimensions to compute global MSE per column
    diff_sq = (preds_selected - targets_selected) ** 2
    mse_per_col = np.mean(diff_sq, axis=(0, 1))
    rmse_per_col = np.sqrt(mse_per_col)

    # 4. Average the RMSEs across the scored columns
    mcrmse = np.mean(rmse_per_col)

    return mcrmse
