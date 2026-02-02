import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Implements the Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function is designed to:
    1. Slice predictions and targets to the scored sequence length (Config.SEQ_SCORED).
    2. Compute the Root Mean Squared Error (RMSE) for each of the 5 target columns independently.
    3. Average the RMSE values across columns to produce a single scalar loss.

    As per the strategy, this loss is computed on all 5 available target columns
    (reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C) to utilize auxiliary
    signals for regularization, rather than restricting optimization to only the
    scored competition metrics.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.seq_scored = Config.SEQ_SCORED

    def forward(self, preds, targets):
        """
        Computes the MCRMSE loss.

        Args:
            preds (torch.Tensor): Model predictions of shape (Batch_Size, Seq_Len, 5).
            targets (torch.Tensor): Ground truth targets of shape (Batch_Size, Seq_Len, 5).

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # Slice predictions and targets to the first 'seq_scored' positions (68)
        # The model outputs 107 positions, but ground truth is only available/valid for the first 68.
        preds_sliced = preds[:, : self.seq_scored, :]
        targets_sliced = targets[:, : self.seq_scored, :]

        # Calculate Mean Squared Error (MSE) for each column independently.
        # We average over the batch (dim 0) and sequence (dim 1) dimensions.
        # Result shape: (5,) - one MSE value per target column.
        mse_per_column = torch.mean((preds_sliced - targets_sliced) ** 2, dim=(0, 1))

        # Calculate Root Mean Squared Error (RMSE) for each column.
        # Result shape: (5,)
        rmse_per_column = torch.sqrt(mse_per_column)

        # Calculate the mean of the column-wise RMSEs to get the final MCRMSE.
        # Result shape: scalar
        loss = torch.mean(rmse_per_column)

        return loss
