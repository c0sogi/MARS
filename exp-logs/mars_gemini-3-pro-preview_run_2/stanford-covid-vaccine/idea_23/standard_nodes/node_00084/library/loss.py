import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) loss,
    strictly masked to the scored sequence positions and target columns.

    This aligns the optimization objective with the competition metric by
    ignoring auxiliary targets (deg_pH10, deg_50C) and unscored sequence positions.
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()
        self.seq_scored = Config.SEQ_SCORED
        self.target_indices = Config.SCORED_TARGET_INDICES
        # Small epsilon for numerical stability in sqrt to prevent NaN gradients if MSE is 0
        self.eps = 1e-6

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Targets).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, Num_Targets).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Slice Sequence Dimension
        # Only the first SEQ_SCORED (68) positions are scored.
        # Inputs/Targets shape becomes: (Batch, 68, Num_Targets)
        inputs_sliced = inputs[:, : self.seq_scored, :]
        targets_sliced = targets[:, : self.seq_scored, :]

        # 2. Select Scored Columns
        # We only optimize for reactivity, deg_Mg_pH10, and deg_Mg_50C.
        # Indices are defined in Config (usually [0, 1, 3]).
        # Shape becomes: (Batch, 68, 3)
        inputs_scored = inputs_sliced[:, :, self.target_indices]
        targets_scored = targets_sliced[:, :, self.target_indices]

        # 3. Compute MSE per column
        # Calculate squared errors: (y - y_hat)^2
        squared_diff = (inputs_scored - targets_scored) ** 2

        # Average over Batch (dim 0) and Sequence (dim 1) to get MSE per column.
        # Result shape: (3,)
        mse_per_col = torch.mean(squared_diff, dim=(0, 1))

        # 4. Compute RMSE per column
        # sqrt(MSE)
        rmse_per_col = torch.sqrt(mse_per_col + self.eps)

        # 5. Average RMSEs (MCRMSE)
        # Final scalar loss
        loss = torch.mean(rmse_per_col)

        return loss
