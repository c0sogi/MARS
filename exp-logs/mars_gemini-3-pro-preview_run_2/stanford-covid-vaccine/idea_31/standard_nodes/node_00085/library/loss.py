import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    Calculates the RMSE for each scored column separately and then averages them.
    This loss function specifically handles the competition requirements by:
    1. Truncating sequences to the scored length (Config.SCORED_LEN).
    2. Selecting only the columns specified in Config.SCORED_COLS.
    """

    def __init__(self):
        super().__init__()
        self.target_cols = Config.TARGET_COLS
        self.scored_cols = Config.SCORED_COLS
        self.scored_len = Config.SCORED_LEN

        # Determine the indices of the columns that contribute to the score
        # e.g., if targets are [A, B, C, D, E] and scored are [A, B, D], indices are [0, 1, 3]
        self.scored_indices = [
            i for i, col in enumerate(self.target_cols) if col in self.scored_cols
        ]

        # Register indices as a buffer if needed, though simple list usage is fine for indexing
        # We convert to a tensor for potential device matching if advanced indexing is used
        self.register_buffer(
            "scored_indices_tensor", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, SeqLen, NumTargets).
            targets (torch.Tensor): Ground truth of shape (Batch, SeqLen, NumTargets).

        Returns:
            torch.Tensor: Scalar MCRMSE loss.
        """
        # 1. Slice to the scored sequence length (ignore positions >= 68)
        # Shape: (Batch, 68, NumTargets)
        inputs_sliced = inputs[:, : self.scored_len, :]
        targets_sliced = targets[:, : self.scored_len, :]

        # 2. Select only the scored columns
        # Shape: (Batch, 68, NumScoredCols)
        # Using the tensor buffer ensures device compatibility
        inputs_scored = torch.index_select(inputs_sliced, 2, self.scored_indices_tensor)
        targets_scored = torch.index_select(
            targets_sliced, 2, self.scored_indices_tensor
        )

        # 3. Calculate MSE per column
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions
        # Result shape: (NumScoredCols,)
        diff = inputs_scored - targets_scored
        mse_per_col = torch.mean(diff**2, dim=(0, 1))

        # 4. Calculate RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # 5. Average the RMSE values across the columns to get MCRMSE
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse
