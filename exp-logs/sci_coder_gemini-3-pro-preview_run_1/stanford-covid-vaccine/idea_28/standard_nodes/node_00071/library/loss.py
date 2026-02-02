import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import PRED_LEN


class MaskedMSELoss(nn.Module):
    """
    Computes the Mean Squared Error (MSE) loss only on the scored positions of the sequence.

    The competition data provides sequences of length 107, but only the first 68 bases
    (defined as PRED_LEN) have valid ground truth labels for scoring. This loss function
    slices the input tensors to ignore the unscored tail of the sequence before calculating MSE.
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.pred_len = PRED_LEN

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds (torch.Tensor): Predicted values. Shape (Batch, Seq_Len, Num_Targets).
            targets (torch.Tensor): Ground truth values. Shape (Batch, Seq_Len, Num_Targets).

        Returns:
            torch.Tensor: Scalar MSE loss computed over the first PRED_LEN positions.
        """
        # Slice inputs to consider only the scored positions
        # Assuming shape is (Batch, Sequence_Length, Channels/Targets)
        preds_scored = preds[:, : self.pred_len, :]
        targets_scored = targets[:, : self.pred_len, :]

        # Compute standard MSE on the valid subset
        loss = F.mse_loss(preds_scored, targets_scored, reduction="mean")

        return loss
