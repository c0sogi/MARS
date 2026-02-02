import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """
    Calculates the Mean Squared Error (MSE) between predictions and targets
    restricted to the valid scored positions (typically the first 68 bases).

    This loss function ignores the unscored tail of the sequence (positions > 68),
    ensuring that gradients are only calculated based on valid experimental data.
    """

    def __init__(self, scored_len=68):
        """
        Args:
            scored_len (int): The number of positions from the start of the sequence
                              that have ground truth values. Default is 68.
        """
        super(MaskedMSELoss, self).__init__()
        self.scored_len = scored_len
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Model predictions of shape (Batch, Seq_Len, Channels).
                                  Typically Seq_Len is 107.
            targets (torch.Tensor): Ground truth targets of shape (Batch, Scored_Len, Channels).
                                    Typically Scored_Len is 68.

        Returns:
            torch.Tensor: Scalar MSE loss calculated over the first `scored_len` positions.
        """
        # Determine the length to score based on targets or configured length
        # We use the target length to ensure shapes match if targets are already sliced
        current_scored_len = targets.shape[1]

        # Slice the predictions to match the target length
        # preds: (B, 107, 3) -> (B, 68, 3)
        preds_sliced = preds[:, :current_scored_len, :]

        # Calculate standard MSE on the sliced tensors
        loss = self.mse(preds_sliced, targets)

        return loss
