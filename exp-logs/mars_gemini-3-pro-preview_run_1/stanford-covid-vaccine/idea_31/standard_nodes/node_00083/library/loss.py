import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MaskedMSELoss(nn.Module):
    """
    Computes Mean Squared Error (MSE) loss restricted to the scored sequence positions.

    The model outputs predictions for the entire sequence length (e.g., 107), but the
    ground truth targets are only provided for the first 'seq_scored' positions (e.g., 68).
    This loss function slices the predictions to align with the targets before calculating MSE.

    It assumes the targets tensor already contains only the valid columns (reactivity,
    deg_Mg_pH10, deg_Mg_50C) as processed by the data pipeline.
    """

    def __init__(self):
        super().__init__()
        self.seq_scored = Config.PRED_LEN

    def forward(self, preds, targets):
        """
        Calculates the masked MSE loss.

        Args:
            preds (torch.Tensor): Model predictions.
                Shape: (Batch, Seq_Len, Num_Targets) -> e.g., (B, 107, 3)
            targets (torch.Tensor): Ground truth values.
                Shape: (Batch, Seq_Scored, Num_Targets) -> e.g., (B, 68, 3)

        Returns:
            torch.Tensor: Scalar MSE loss value.
        """
        # Slice the predictions to match the length of the available ground truth
        # We only care about the first 'seq_scored' positions (0 to 67)
        preds_sliced = preds[:, : self.seq_scored, :]

        # Ensure that the targets provided match the expected scored length
        # (This is implicitly guaranteed by the dataset class, but good for safety)
        if targets.shape[1] != self.seq_scored:
            raise ValueError(
                f"Target sequence length {targets.shape[1]} does not match "
                f"configured scored length {self.seq_scored}."
            )

        # Compute standard Mean Squared Error
        # F.mse_loss defaults to reduction='mean'
        loss = F.mse_loss(preds_sliced, targets)

        return loss
