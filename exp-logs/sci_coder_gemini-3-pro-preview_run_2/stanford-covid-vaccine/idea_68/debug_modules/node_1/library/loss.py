import torch
import torch.nn as nn
from library import config


class MaskedMCRMSELoss(nn.Module):
    """
    Custom Loss function for RNA Degradation Prediction.

    Implements Mean Columnwise Root Mean Squared Error (MCRMSE) with strict masking:
    1. Column Masking: Only calculates loss for scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    2. Sequence Masking: Only calculates loss for the first 68 bases (seq_scored), ignoring the tail.

    This prevents the model from optimizing on unscored targets or zero-padded regions,
    avoiding multi-task conflicts and ensuring physical consistency.
    """

    def __init__(self):
        super().__init__()
        # Indices corresponding to ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
        # Based on ALL_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        self.scored_col_indices = [0, 1, 3]
        self.scored_len = config.SCORED_LEN

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): Model predictions of shape (Batch, 5, Seq_Len).
            targets (torch.Tensor): Ground truth targets of shape (Batch, 5, Seq_Len).

        Returns:
            torch.Tensor: Scalar MCRMSE loss.
        """
        # 1. Slice to keep only scored columns: (B, 3, L)
        preds_cols = preds[:, self.scored_col_indices, :]
        targets_cols = targets[:, self.scored_col_indices, :]

        # 2. Slice to keep only scored sequence positions: (B, 3, 68)
        # We ignore positions > 67 as they are not scored and often zero-padded
        preds_masked = preds_cols[:, :, : self.scored_len]
        targets_masked = targets_cols[:, :, : self.scored_len]

        # 3. Calculate MSE
        mse = (preds_masked - targets_masked) ** 2

        # 4. Calculate RMSE per column
        # Average over Batch (dim 0) and Sequence Length (dim 2)
        # Result shape: (3,)
        mse_per_col = torch.mean(mse, dim=(0, 2))
        rmse_per_col = torch.sqrt(mse_per_col)

        # 5. Average RMSEs across columns to get MCRMSE
        loss = torch.mean(rmse_per_col)

        return loss
