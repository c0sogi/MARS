import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Implements the Masked Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function enforces strict masking protocols as defined in the HC-SDRN strategy:
    1. Sequence Masking: Limits calculation to the first `pred_len` positions (e.g., 68),
       ignoring the zero-padded tail to prevent gradient conflicts from artificial zeros.
    2. Target Masking: Limits calculation to specific `scored_indices` (e.g., reactivity,
       deg_Mg_pH10, deg_Mg_50C), ignoring unscored columns to prevent unsupervised noise injection.
    """

    def __init__(self, scored_indices=None, pred_len=None):
        """
        Args:
            scored_indices (list, optional): Indices of the columns to score.
                                             Defaults to Config.SCORED_INDICES.
            pred_len (int, optional): The length of the sequence to score.
                                      Defaults to Config.PRED_LEN.
        """
        super().__init__()
        self.scored_indices = (
            scored_indices if scored_indices is not None else Config.SCORED_INDICES
        )
        self.pred_len = pred_len if pred_len is not None else Config.PRED_LEN

    def forward(self, pred, target):
        """
        Calculates the MCRMSE loss between predictions and targets.

        Args:
            pred (torch.Tensor): Predicted values of shape (Batch, Seq_Len, 5).
            target (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, 5).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Strict Sequence Masking
        # Slice to the valid scored length (e.g., indices 0-67)
        # We assume the inputs are (Batch, Seq_Len, Channels)
        pred_scored = pred[:, : self.pred_len, :]
        target_scored = target[:, : self.pred_len, :]

        # 2. Strict Target Masking
        # Select only the scored columns (e.g., indices 0, 1, 3)
        pred_scored = pred_scored[:, :, self.scored_indices]
        target_scored = target_scored[:, :, self.scored_indices]

        # 3. Compute MSE per column
        # Flatten batch and sequence dimensions to compute global MSE per column for the batch
        # Shape becomes: (Batch * Pred_Len, Num_Scored_Cols)
        pred_flat = pred_scored.reshape(-1, len(self.scored_indices))
        target_flat = target_scored.reshape(-1, len(self.scored_indices))

        # Mean Squared Error per column
        mse_cols = torch.mean((pred_flat - target_flat) ** 2, dim=0)

        # 4. Root Mean Squared Error per column
        rmse_cols = torch.sqrt(mse_cols)

        # 5. Average across columns (MCRMSE)
        loss = torch.mean(rmse_cols)

        return loss
