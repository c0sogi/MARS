import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Column-wise Root Mean Squared Error Loss.

    Calculates the RMSE for each target column separately and averages them.
    This corresponds to the MCRMSE metric used in the competition, applied
    as a loss function during training.

    It handles the length mismatch between model output (107) and targets (68)
    by slicing the predictions.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions of shape (batch_size, seq_len, num_targets).
                                   Typically seq_len is 107.
            targets (torch.Tensor): Ground truth of shape (batch_size, pred_len, num_targets).
                                    Typically pred_len is 68.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Slice inputs to match the length of targets (first 68 positions)
        # inputs: (B, 107, 5) -> (B, 68, 5)
        inputs_sliced = inputs[:, : Config.PRED_LEN, :]

        # Calculate MSE per element: (B, 68, 5)
        mse_per_element = self.mse(inputs_sliced, targets)

        # Average over batch and sequence dimensions to get MSE per column: (5,)
        # We want the average MSE for each of the 5 target types
        mse_per_column = torch.mean(mse_per_element, dim=(0, 1))

        # Take square root to get RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Average the RMSEs across the columns to get the final loss
        loss = torch.mean(rmse_per_column)

        return loss


def competition_metric(preds, targets):
    """
    Calculates the official competition metric (MCRMSE) on the scored columns.

    This function:
    1. Slices predictions to the scored sequence length (68).
    2. Filters for the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.
    3. Computes RMSE globally (across the entire provided set) for each column.
    4. Returns the mean of these RMSEs.

    Args:
        preds (torch.Tensor or np.ndarray): Predictions of shape (N, 107, 5).
        targets (torch.Tensor or np.ndarray): Ground truth of shape (N, 68, 5).

    Returns:
        float: The MCRMSE score.
    """
    # Convert to tensor if numpy array
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)

    # Ensure on CPU for calculation to avoid GPU memory overhead on large validation sets
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    # 1. Slice predictions to match target length (68)
    preds_sliced = preds[:, : Config.PRED_LEN, :]

    # 2. Select only the scored columns
    # Config.SCORED_COLS_INDICES is [0, 1, 3] corresponding to
    # reactivity, deg_Mg_pH10, deg_Mg_50C
    preds_scored = preds_sliced[:, :, Config.SCORED_COLS_INDICES]
    targets_scored = targets[:, :, Config.SCORED_COLS_INDICES]

    # 3. Calculate MSE globally for each column
    # (N, 68, 3) -> (N * 68, 3) -> Mean over dim 0 -> (3,)
    mse_per_column = torch.mean((preds_scored - targets_scored) ** 2, dim=(0, 1))

    # 4. Calculate RMSE per column
    rmse_per_column = torch.sqrt(mse_per_column)

    # 5. Average RMSEs
    mcrmse = torch.mean(rmse_per_column)

    return mcrmse.item()
