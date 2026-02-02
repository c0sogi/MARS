import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function is specific to the RNA degradation task. It:
    1. Selects only the scored columns defined in Config.SCORED_TARGETS.
    2. Selects only the scored sequence positions (0 to Config.PRED_LEN).
    3. Computes RMSE for each column separately.
    4. Returns the mean of these RMSE values.
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()
        self.scored_targets = Config.SCORED_TARGETS
        self.pred_len = Config.PRED_LEN

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predictions of shape (Batch, 5, Seq_Len).
            target (torch.Tensor): Ground truth of shape (Batch, Seq_Len, 5).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Align target shape to (Batch, 5, Seq_Len) to match predictions
        if target.shape[1] != 5 and target.shape[2] == 5:
            target = target.permute(0, 2, 1)

        # Select only the scored target columns
        # Indices: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
        pred_scored = pred[:, self.scored_targets, :]
        target_scored = target[:, self.scored_targets, :]

        # Select only the scored sequence positions (first 68 bases)
        pred_scored = pred_scored[:, :, : self.pred_len]
        target_scored = target_scored[:, :, : self.pred_len]

        # Calculate Squared Error: (Batch, Num_Scored_Cols, Pred_Len)
        mse = (pred_scored - target_scored) ** 2

        # Average over Batch (dim 0) and Sequence Length (dim 2) to get MSE per column
        # Result shape: (Num_Scored_Cols,)
        # We add a small epsilon to the denominator for numerical stability, though usually not needed with mean
        column_mse = torch.mean(mse, dim=(0, 2))

        # Calculate RMSE per column
        column_rmse = torch.sqrt(column_mse)

        # Calculate the Mean of the column RMSEs (MCRMSE)
        loss = torch.mean(column_rmse)

        return loss
