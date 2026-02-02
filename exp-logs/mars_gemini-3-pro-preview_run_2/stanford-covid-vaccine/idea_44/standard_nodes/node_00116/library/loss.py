import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) with strict masking.

    This loss function ensures that:
    1. Only the first `seq_scored` positions (0 to 67) are used for calculation.
    2. Only the specific `scored_cols` (reactivity, deg_Mg_pH10, deg_Mg_50C) contribute to the loss.

    Formula:
        MCRMSE = Mean( RMSE(col_1), RMSE(col_2), ... )
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()
        self.seq_scored = Config.SEQ_SCORED
        self.scored_indices = Config.SCORED_INDICES

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted values. Shape (Batch, Seq_Len, 5).
            targets (torch.Tensor): Ground truth values. Shape (Batch, Seq_Len, 5) or (Batch, Seq_Scored, 5).

        Returns:
            torch.Tensor: A scalar loss value.
        """
        # 1. Slice Sequence Dimension
        # We only care about the first `seq_scored` bases.
        # Ensure we don't exceed the dimensions of the tensors.
        inputs_sliced = inputs[:, : self.seq_scored, :]
        targets_sliced = targets[:, : self.seq_scored, :]

        # 2. Filter Scored Columns
        # We select only the indices corresponding to the scored targets.
        # inputs_filtered shape: (Batch, Seq_Scored, 3)
        inputs_filtered = inputs_sliced[:, :, self.scored_indices]
        targets_filtered = targets_sliced[:, :, self.scored_indices]

        # 3. Compute Squared Error
        squared_diff = (inputs_filtered - targets_filtered) ** 2

        # 4. Compute MSE per column
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions.
        # Result shape: (3,) corresponding to the 3 scored columns.
        mse_per_col = torch.mean(squared_diff, dim=(0, 1))

        # 5. Compute RMSE per column
        # Add a small epsilon for numerical stability if needed, though usually not strictly necessary with MSE
        rmse_per_col = torch.sqrt(mse_per_col)

        # 6. Compute Mean of RMSEs (MCRMSE)
        loss = torch.mean(rmse_per_col)

        return loss
