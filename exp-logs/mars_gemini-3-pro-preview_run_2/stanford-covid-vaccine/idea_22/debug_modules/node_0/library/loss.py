import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss,
    specifically masked to evaluate only the scored target columns and
    valid sequence positions defined in the competition configuration.
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()
        self.scored_indices = Config.SCORED_INDICES
        self.pred_len = Config.PRED_LEN

    def forward(self, inputs, targets):
        """
        Compute the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Model predictions with shape (Batch, Seq_Len, Num_Targets).
                                   Usually Seq_Len is 107.
            targets (torch.Tensor): Ground truth targets with shape (Batch, Scored_Seq_Len, Num_Targets).
                                    Usually Scored_Seq_Len is 68.

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # 1. Slice predictions to match the scored sequence length (first 68 positions)
        # inputs shape: (B, 107, 5) -> (B, 68, 5)
        if inputs.shape[1] > self.pred_len:
            inputs = inputs[:, : self.pred_len, :]

        # 2. Select only the columns that contribute to the score
        # indices: [0, 1, 3] corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C
        inputs_scored = inputs[:, :, self.scored_indices]
        targets_scored = targets[:, :, self.scored_indices]

        # 3. Calculate Mean Squared Error (MSE) for each column independently
        # We average over Batch (dim 0) and Sequence (dim 1) dimensions
        mse = torch.mean((inputs_scored - targets_scored) ** 2, dim=(0, 1))

        # 4. Calculate Root Mean Squared Error (RMSE) for each column
        rmse = torch.sqrt(mse)

        # 5. Calculate the Mean of the RMSEs (MCRMSE)
        loss = torch.mean(rmse)

        return loss
