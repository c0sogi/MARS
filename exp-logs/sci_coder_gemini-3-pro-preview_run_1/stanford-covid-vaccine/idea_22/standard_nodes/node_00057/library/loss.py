import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MaskedMSELoss(nn.Module):
    """
    Computes the Mean Squared Error (MSE) loss only for the valid scored positions.

    According to the task description, only the first 68 bases (defined in Config.PRED_LEN)
    are scored. The model outputs predictions for the full sequence length (107),
    but the loss should strictly focus on the scored region to avoid training on
    unverified or missing data in the tail.
    """

    def __init__(self):
        super().__init__()
        self.pred_len = Config.PRED_LEN

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions of shape (Batch, Seq_Len, Num_Targets).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Num_Targets).

        Returns:
            torch.Tensor: Scalar MSE loss computed over the first `pred_len` positions.
        """
        # Slice the inputs and targets to include only the first 68 positions
        # Shape becomes: (Batch, 68, Num_Targets)
        masked_inputs = inputs[:, : self.pred_len, :]
        masked_targets = targets[:, : self.pred_len, :]

        # Compute Mean Squared Error
        # We use the default reduction='mean' to get the average error
        loss = F.mse_loss(masked_inputs, masked_targets)

        return loss
