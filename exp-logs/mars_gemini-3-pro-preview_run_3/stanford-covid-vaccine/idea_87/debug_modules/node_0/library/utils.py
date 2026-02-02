import os
import random
import numpy as np
import torch
import torch.nn as nn
from library.config import Config


def set_seed(seed=Config.SEED):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
    Ensures deterministic behavior for CuDNN backend.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error Loss.

    Implements the Multi-Task Learning objective:
    - Calculates RMSE for each of the 5 target columns.
    - Averages the RMSEs to produce the final loss.
    - Slices predictions to the first 68 positions (Config.PRED_LEN) to match ground truth.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, preds, targets):
        """
        Args:
            preds: (Batch, Seq_Len, 5) - Model predictions (likely Seq_Len=107).
            targets: (Batch, Pred_Len, 5) - Ground truth targets (Pred_Len=68).

        Returns:
            loss: Scalar MCRMSE loss averaged over all 5 columns.
        """
        # Slice predictions to the scored sequence length (68)
        # We use the length of the targets to determine the slice point dynamically
        pred_len = targets.shape[1]
        preds_sliced = preds[:, :pred_len, :]

        # Calculate MSE per element: (Batch, Pred_Len, 5)
        mse = self.mse(preds_sliced, targets)

        # Average over batch and sequence length to get MSE per column: (5,)
        mse_per_col = torch.mean(mse, dim=(0, 1))

        # RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Average RMSE across all 5 columns (Multi-Task Learning)
        loss = torch.mean(rmse_per_col)

        return loss


def metric_mcrmse(preds, targets):
    """
    Calculates the MCRMSE metric specifically for the 3 scored columns
    on the first 68 positions.

    This function strictly adheres to the competition scoring:
    1. Slices data to 'seq_scored' (68).
    2. Selects only ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'].

    Args:
        preds: (Batch, Seq_Len, 5) or (Batch, Pred_Len, 5) - Model predictions.
        targets: (Batch, Pred_Len, 5) - Ground truth.

    Returns:
        mcrmse: Scalar float, the calculated metric.
    """
    # Ensure inputs are numpy arrays
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Slice predictions to match target length (68)
    pred_len = targets.shape[1]
    preds_sliced = preds[:, :pred_len, :]

    # Identify indices of scored columns based on Config
    # Config.TARGET_COLS contains all 5 targets
    # Config.SCORED_COLS contains the 3 scored targets
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS

    # Find indices of scored columns within the target columns list
    # Typically indices [0, 1, 3] corresponding to ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    # Filter for scored columns
    preds_scored = preds_sliced[:, :, scored_indices]
    targets_scored = targets[:, :, scored_indices]

    # Calculate MSE per column
    # Mean over Batch (axis 0) and Sequence (axis 1)
    mse_per_col = np.mean((preds_scored - targets_scored) ** 2, axis=(0, 1))

    # RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Mean of RMSEs
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
