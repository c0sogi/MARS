import torch
import torch.nn as nn
from library import config


class MaskedMCRMSELoss(nn.Module):
    """
    Custom Loss function that calculates the Mean Columnwise Root Mean Squared Error (MCRMSE)
    strictly on valid scored sequence positions and scored target columns.

    This implementation:
    1. Masks out sequence positions beyond config.SEQ_SCORED (0-67 used, 68+ ignored).
    2. Masks out target columns not in config.SCORED_COLS_INDICES.
    3. Computes RMSE for each remaining column globally across the batch and sequence.
    4. Returns the average of these RMSE values.
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()
        self.seq_scored = config.SEQ_SCORED
        # Register indices as a buffer so it moves to device automatically if part of state_dict,
        # though here we handle device manually in forward to be safe.
        self.register_buffer(
            "scored_indices", torch.tensor(config.SCORED_COLS_INDICES, dtype=torch.long)
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the masked MCRMSE loss.

        Args:
            preds (torch.Tensor): Model predictions with shape (Batch, Seq_Len, 5).
            targets (torch.Tensor): Ground truth targets with shape (Batch, Seq_Len, 5).

        Returns:
            torch.Tensor: A scalar tensor representing the loss.
        """
        # 1. Strict Sequence Masking
        # Select only the first `seq_scored` positions (e.g., 0 to 67).
        # We ignore the zero-padded tail to prevent gradient contamination from artificial zeros.
        preds_valid = preds[:, : self.seq_scored, :]
        targets_valid = targets[:, : self.seq_scored, :]

        # 2. Strict Target Column Masking
        # Select only the columns that are actually scored in the competition.
        # Typically: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3).
        # We use index_select for robust column extraction.
        preds_selected = torch.index_select(preds_valid, 2, self.scored_indices)
        targets_selected = torch.index_select(targets_valid, 2, self.scored_indices)

        # 3. Compute MSE per column
        # We calculate the Mean Squared Error for each column independently,
        # averaging over both the Batch dimension (0) and the Sequence dimension (1).
        mse_per_col = torch.mean((preds_selected - targets_selected) ** 2, dim=(0, 1))

        # 4. Compute RMSE per column
        # Add a small epsilon for numerical stability if needed, though usually not strictly necessary with MSE.
        rmse_per_col = torch.sqrt(mse_per_col)

        # 5. Compute MCRMSE
        # Average the RMSE values across the scored columns.
        loss = torch.mean(rmse_per_col)

        return loss
