import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function applies strict masking:
    1. Sequence Masking: Only considers the first `Config.SCORED_LEN` positions (0-67).
       The remaining positions (68-106) are ignored as they are not scored.
    2. Column Masking: Only considers the scored target columns specified in
       `Config.SCORED_TARGET_INDICES` (reactivity, deg_Mg_pH10, deg_Mg_50C).

    Formula:
        MCRMSE = (1/Nt) * Sum_j( sqrt( (1/n) * Sum_i( (y_ij - y_hat_ij)^2 ) ) )
        where Nt is number of scored columns, n is total valid elements (Batch * Scored_Len).
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.scored_len = Config.SCORED_LEN
        self.scored_cols = Config.SCORED_TARGET_INDICES

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, 5).

        Returns:
            torch.Tensor: Scalar MCRMSE loss.
        """
        # 1. Slice to valid sequence length (e.g., first 68 bases)
        # Shape becomes: (Batch, 68, 5)
        pred_valid = inputs[:, : self.scored_len, :]
        target_valid = targets[:, : self.scored_len, :]

        # 2. Slice to scored columns (e.g., indices 0, 1, 3)
        # Shape becomes: (Batch, 68, 3)
        pred_scored = pred_valid[:, :, self.scored_cols]
        target_scored = target_valid[:, :, self.scored_cols]

        # 3. Compute MSE for each column independently
        # We average over the batch (dim 0) and sequence (dim 1) dimensions.
        # Result shape: (3,) - one MSE value per scored column.
        column_mse = torch.mean((pred_scored - target_scored) ** 2, dim=(0, 1))

        # 4. Compute RMSE for each column
        column_rmse = torch.sqrt(column_mse)

        # 5. Average the RMSEs across columns to get MCRMSE
        loss = torch.mean(column_rmse)

        return loss
