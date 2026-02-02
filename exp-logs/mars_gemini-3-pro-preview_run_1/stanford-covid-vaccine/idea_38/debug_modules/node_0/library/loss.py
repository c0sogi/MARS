import torch
import torch.nn as nn
from library.config import Config


class MaskedMSELoss(nn.Module):
    """
    Calculates the Mean Squared Error (MSE) loss restricted to the scored sequence positions.

    As per the competition requirements:
    - Loss is calculated only for the first `seq_scored` (68) positions.
    - The unscored tail (positions 68-106) is ignored during training.
    - Standard MSE is used without sample weighting.
    """

    def __init__(self, seq_scored=Config.SEQ_SCORED):
        """
        Initialize the MaskedMSELoss.

        Args:
            seq_scored (int): The number of positions from the start of the sequence
                              to include in the loss calculation. Defaults to Config.SEQ_SCORED (68).
        """
        super(MaskedMSELoss, self).__init__()
        self.seq_scored = seq_scored
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        """
        Compute the masked MSE loss.

        Args:
            preds (torch.Tensor): Model predictions.
                                  Expected shape: (Batch, Seq_Len, Num_Targets)
                                  typically (B, 107, 3).
            targets (torch.Tensor): Ground truth values.
                                    Expected shape: (Batch, Seq_Len_Targets, Num_Targets)
                                    typically (B, 68, 3) or (B, 107, 3).

        Returns:
            torch.Tensor: Scalar MSE loss computed over the first `seq_scored` positions.
        """
        # Slice predictions to the scored length
        # If preds are longer than seq_scored (e.g., 107), take the first 68.
        if preds.shape[1] > self.seq_scored:
            preds_sliced = preds[:, : self.seq_scored, :]
        else:
            preds_sliced = preds

        # Slice targets to the scored length
        # If targets are provided with full length (e.g., padded), take the first 68.
        if targets.shape[1] > self.seq_scored:
            targets_sliced = targets[:, : self.seq_scored, :]
        else:
            targets_sliced = targets

        # Ensure shapes match before loss calculation to prevent silent broadcasting errors
        if preds_sliced.shape != targets_sliced.shape:
            raise ValueError(
                f"Shape mismatch after slicing in MaskedMSELoss: "
                f"Preds {preds_sliced.shape} vs Targets {targets_sliced.shape}"
            )

        return self.mse(preds_sliced, targets_sliced)
