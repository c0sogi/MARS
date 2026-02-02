import torch
import torch.nn as nn
from library.config import Config


class MaskedMSELoss(nn.Module):
    """
    Computes the Mean Squared Error (MSE) loss restricted to the scored sequence positions.

    The RNA degradation task evaluates predictions only on the first `seq_scored` positions
    (typically 68), ignoring the rest of the sequence (length 107). This loss function
    slices the model predictions and the ground truth targets to the scoring length before
    computing the standard MSE.
    """

    def __init__(self, scoring_length=Config.PRED_LEN):
        """
        Initialize the MaskedMSELoss.

        Args:
            scoring_length (int): The number of positions from the start of the sequence
                                  to include in the loss calculation. Defaults to Config.PRED_LEN (68).
        """
        super(MaskedMSELoss, self).__init__()
        self.scoring_length = scoring_length
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        """
        Compute the masked MSE loss.

        Args:
            inputs (torch.Tensor): Model predictions. Shape (Batch, Seq_Len, Channels).
                                   Typically (Batch, 107, 3).
            targets (torch.Tensor): Ground truth values. Shape (Batch, Seq_Len, Channels)
                                    or (Batch, Scored_Len, Channels).

        Returns:
            torch.Tensor: Scalar MSE loss computed over the first `scoring_length` positions.
        """
        # Slice predictions to the scored length
        # inputs shape: [B, 107, 3] -> [B, 68, 3]
        masked_inputs = inputs[:, : self.scoring_length, :]

        # Slice targets to the scored length
        # This safely handles cases where targets might be padded to the full sequence length
        # or provided as just the scored length.
        masked_targets = targets[:, : self.scoring_length, :]

        return self.mse(masked_inputs, masked_targets)
