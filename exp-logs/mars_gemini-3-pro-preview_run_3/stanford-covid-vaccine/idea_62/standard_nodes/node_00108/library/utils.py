import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Column-wise Root Mean Squared Error (MCRMSE) Loss.

    Formula:
    MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Where:
    - j iterates over target columns (Nt)
    - i iterates over samples and sequence positions (n)

    As per the strategy, this loss calculates the metric over ALL 5 target columns
    to utilize auxiliary signals during training.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Model predictions of shape (Batch, Seq_Len_Pred, Num_Targets)
                                  typically (B, 107, 5).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len_Target, Num_Targets)
                                    typically (B, 68, 5).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Slice predictions to match target sequence length (usually 68)
        # The model outputs 107 positions, but we only have ground truth for the first 68.
        seq_scored = targets.shape[1]
        preds_sliced = preds[:, :seq_scored, :]

        # Calculate MSE for each element
        mse_loss = self.mse(preds_sliced, targets)

        # Average over batch and sequence dimensions (dim 0 and 1) to get MSE per column
        # shape: (Num_Targets,)
        column_mse = torch.mean(mse_loss, dim=(0, 1))

        # Take square root to get RMSE per column
        column_rmse = torch.sqrt(column_mse)

        # Average over columns to get MCRMSE
        loss = torch.mean(column_rmse)

        return loss


def get_scored_col_indices():
    """
    Helper to find indices of the scored columns within the full target list.
    """
    return [Config.TARGET_COLS.index(col) for col in Config.SCORED_COLS]


def calculate_metric(preds, targets):
    """
    Calculates the MCRMSE metric specifically for the validation set.

    According to the task description and strategy:
    1. Slices data to the first 68 positions (`seq_scored`).
    2. Filters only the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.

    Args:
        preds (np.ndarray or torch.Tensor): Predictions (B, 107, 5) or (B, 68, 5).
        targets (np.ndarray or torch.Tensor): Ground truth (B, 68, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Convert to numpy if tensors
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # 1. Slice predictions to seq_scored (68)
    # Ensure we don't slice if it's already the correct length, but usually preds are 107.
    seq_scored = Config.SEQ_SCORED
    if preds.shape[1] > seq_scored:
        preds = preds[:, :seq_scored, :]

    # Verify shapes match on sequence length now
    assert (
        preds.shape[1] == targets.shape[1]
    ), f"Sequence length mismatch after slicing: Preds {preds.shape[1]}, Targets {targets.shape[1]}"

    # 2. Filter for scored columns
    scored_indices = get_scored_col_indices()

    preds_scored = preds[:, :, scored_indices]
    targets_scored = targets[:, :, scored_indices]

    # 3. Calculate MCRMSE
    # MSE per column: Mean over batch (0) and sequence (1)
    mse_per_col = np.mean((preds_scored - targets_scored) ** 2, axis=(0, 1))

    # RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Mean of RMSEs
    mcrmse = np.mean(rmse_per_col)

    return float(mcrmse)
