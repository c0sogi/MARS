import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MaskedMSELoss(nn.Module):
    """
    Implements the Masked Mean Squared Error (MSE) loss function.

    According to the 'Stabilized High-Capacity Wide-Stream BiLSTM' strategy:
    1. Calculates loss ONLY for the first 68 positions (Config.SCORED_LEN).
    2. Strictly uses MSE (L2) to align with the RMSE evaluation metric.
    3. Ignores unscored positions (padding) and treats all samples equally.
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.scored_len = Config.SCORED_LEN

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the MSE loss on the valid scored positions.

        Args:
            inputs (torch.Tensor): Model predictions of shape (Batch, Seq_Len, Num_Targets).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Num_Targets).

        Returns:
            torch.Tensor: Scalar MSE loss value.
        """
        # Slice the tensors to include only the scored positions (0 to 67)
        # inputs shape becomes: (Batch, 68, Num_Targets)
        inputs_scored = inputs[:, : self.scored_len, :]

        # targets shape becomes: (Batch, 68, Num_Targets)
        targets_scored = targets[:, : self.scored_len, :]

        # Compute standard Mean Squared Error
        loss = F.mse_loss(inputs_scored, targets_scored)

        return loss
