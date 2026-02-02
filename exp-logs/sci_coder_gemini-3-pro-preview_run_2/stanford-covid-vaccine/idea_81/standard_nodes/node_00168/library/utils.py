import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Column-wise Root Mean Squared Error Loss.

    This loss function calculates the RMSE for each target column separately
    and then averages them. It is designed to work with the 'Anchored' strategy,
    calculating loss over the full sequence length (e.g., 107) where the
    tail (68-107) targets are filled with 0.0 to anchor the model state.
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor of shape (Batch, SeqLen, NumTargets)
            targets: Tensor of shape (Batch, SeqLen, NumTargets)

        Returns:
            loss: Scalar tensor representing the mean MCRMSE.
        """
        # Calculate MSE for each column, averaging over Batch and SeqLen dimensions
        # preds and targets are expected to be (B, 107, 5)
        mse = torch.mean((preds - targets) ** 2, dim=(0, 1))

        # Calculate RMSE for each column. Add epsilon for numerical stability.
        rmse = torch.sqrt(mse + 1e-8)

        # Average the RMSEs across the target columns
        loss = torch.mean(rmse)

        return loss


def compute_global_rmse(preds, targets):
    """
    Computes the global MCRMSE metric for validation on the scored positions.

    Logic:
    1. Slices data to the first `Config.PRED_LEN` (68) positions.
    2. Filters for the specific scored columns defined in `Config.SCORED_COLS`.
    3. Computes RMSE globally (aggregating errors across all samples) rather than
       averaging per-sample RMSE, consistent with the competition metric.

    Args:
        preds: Numpy array of shape (N_samples, SeqLen, NumTargets)
        targets: Numpy array of shape (N_samples, SeqLen, NumTargets)

    Returns:
        final_metric: float, the mean column-wise RMSE over scored columns.
        col_metrics: dict, RMSE per scored column.
    """
    # Determine indices of scored columns
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    scored_indices = [
        i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
    ]

    # Slice to scored sequence length (e.g., 68)
    preds_scored = preds[:, : Config.PRED_LEN, :]
    targets_scored = targets[:, : Config.PRED_LEN, :]

    # Select only the scored columns
    preds_filtered = preds_scored[:, :, scored_indices]
    targets_filtered = targets_scored[:, :, scored_indices]

    # Compute squared errors
    squared_errors = (preds_filtered - targets_filtered) ** 2

    # Compute MSE per column (aggregating over samples and sequence positions)
    # axis=(0, 1) collapses (N_samples, SeqLen)
    mse_per_col = np.mean(squared_errors, axis=(0, 1))

    # Compute RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Compute the final metric (mean of column RMSEs)
    final_metric = np.mean(rmse_per_col)

    # Create dictionary for individual column metrics
    col_metrics = {
        col_name: rmse for col_name, rmse in zip(Config.SCORED_COLS, rmse_per_col)
    }

    return final_metric, col_metrics
