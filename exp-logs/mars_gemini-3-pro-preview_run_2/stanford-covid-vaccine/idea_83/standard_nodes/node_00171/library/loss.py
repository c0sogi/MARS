import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSE(nn.Module):
    """
    Masked Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function strictly enforces two types of masking as per the competition requirements:
    1. Sequence Masking: Only the first 'seq_scored' positions (typically 68) are considered.
       Positions beyond this (the zero-padded tail) are ignored to prevent learning from artifacts.
    2. Column Masking: Only the scored target columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
       are considered. Unscored columns are ignored.
    """

    def __init__(self, mask_sequence=False):
        super(MaskedMCRMSE, self).__init__()
        self.scored_seq_len = Config.SCORED_SEQ_LEN
        self.scored_target_indices = Config.SCORED_TARGET_INDICES
        self.mask_sequence = mask_sequence

    def forward(self, pred, target):
        """
        Compute the masked MCRMSE loss.

        Args:
            pred (torch.Tensor): Predicted values. Shape (Batch, Seq_Len, 5).
            target (torch.Tensor): Ground truth values. Shape (Batch, Seq_Len, 5).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Sequence Masking (Optional)
        # Cite Lesson 00136: Do not mask loss in unmeasured regions for BiRNNs; anchor with zeros.
        if self.mask_sequence:
            current_seq_len = pred.shape[1]
            valid_len = min(current_seq_len, self.scored_seq_len)
            pred_valid = pred[:, :valid_len, :]
            target_valid = target[:, :valid_len, :]
        else:
            pred_valid = pred
            target_valid = target

        # 2. Column Masking
        # We select only the specific columns that are scored.
        # Config.SCORED_TARGET_INDICES usually contains [0, 1, 3] corresponding to
        # reactivity, deg_Mg_pH10, and deg_Mg_50C.
        pred_scored = pred_valid[:, :, self.scored_target_indices]
        target_scored = target_valid[:, :, self.scored_target_indices]

        # 3. Compute MSE per column
        # We calculate the mean squared error across the Batch (dim 0) and Sequence (dim 1) dimensions.
        # We keep the Channel dimension (dim 2) separate to calculate column-wise metrics first.
        mse_per_col = torch.mean((pred_scored - target_scored) ** 2, dim=(0, 1))

        # 4. Compute RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # 5. Compute MCRMSE (Mean of column-wise RMSEs)
        # We average the RMSE values across the selected columns.
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse
