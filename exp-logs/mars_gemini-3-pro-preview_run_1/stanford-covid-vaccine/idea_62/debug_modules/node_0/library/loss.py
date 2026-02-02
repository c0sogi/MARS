import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class MaskedMSELoss(nn.Module):
    """
    Computes Mean Squared Error restricted to the first `seq_scored` positions.
    This aligns with the availability of ground truth data.
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.seq_scored = Config.SEQ_SCORED
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        """
        Args:
            preds: Predictions of shape (batch_size, seq_len, num_targets)
            targets: Ground truth of shape (batch_size, seq_len, num_targets)
                     or (batch_size, seq_scored, num_targets)
        Returns:
            torch.Tensor: Scalar MSE loss computed over the first `seq_scored` positions.
        """
        # Slice predictions to the scored sequence length (e.g., first 68 bases)
        preds_scored = preds[:, : self.seq_scored, :]

        # Slice targets if they are provided with full sequence length
        if targets.shape[1] > self.seq_scored:
            targets_scored = targets[:, : self.seq_scored, :]
        else:
            targets_scored = targets

        return self.mse(preds_scored, targets_scored)


def mcrmse(preds, targets):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Logic:
    1. Slice data to the scored region (first `seq_scored` positions).
    2. Compute RMSE for each target column separately.
    3. Average the RMSE values across columns.

    Args:
        preds (torch.Tensor or np.ndarray): Model predictions.
        targets (torch.Tensor or np.ndarray): Ground truth values.

    Returns:
        float: The MCRMSE score.
    """
    seq_scored = Config.SEQ_SCORED

    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Slice predictions to the scored region
    preds = preds[:, :seq_scored, :]

    # Slice targets to the scored region if necessary
    if targets.shape[1] > seq_scored:
        targets = targets[:, :seq_scored, :]

    # Calculate Squared Error
    squared_diff = (preds - targets) ** 2

    # Calculate MSE per column: Average over Batch (axis 0) and Sequence (axis 1)
    # Result shape: (num_targets,)
    mse_per_col = np.mean(squared_diff, axis=(0, 1))

    # Calculate RMSE per column
    rmse_per_col = np.sqrt(mse_per_col)

    # Final Metric: Mean of the column-wise RMSEs
    return float(np.mean(rmse_per_col))
