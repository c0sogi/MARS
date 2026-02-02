import torch
import torch.nn as nn
from library.config import Config


class MaskedMSELoss(nn.Module):
    """
    Computes Mean Squared Error (MSE) loss restricted to the first `seq_scored` positions.

    The RNA degradation task only scores the first 68 nucleotides (PRED_LEN),
    even though the sequence length is 107. This loss function ensures that
    gradients are only calculated for the valid scored positions, ignoring the
    unscored tail where experimental data is missing or unreliable.
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.pred_len = Config.PRED_LEN
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions from the model.
                                   Shape: (Batch, Seq_Len, Channels) e.g., (B, 107, 3)
            targets (torch.Tensor): Ground truth values.
                                    Shape: (Batch, Target_Len, Channels) e.g., (B, 68, 3) or (B, 107, 3)

        Returns:
            torch.Tensor: Scalar MSE loss computed over the first `pred_len` positions.
        """
        # Slice the inputs to consider only the scored positions
        # We take the first 68 positions along the sequence dimension (dim 1)
        masked_inputs = inputs[:, : self.pred_len, :]

        # Slice targets to match dimensions.
        # Even if targets are already length 68, this slice is safe.
        # If targets were padded to 107, this removes the padding/unscored data.
        masked_targets = targets[:, : self.pred_len, :]

        # Compute standard MSE on the sliced tensors
        return self.mse(masked_inputs, masked_targets)
