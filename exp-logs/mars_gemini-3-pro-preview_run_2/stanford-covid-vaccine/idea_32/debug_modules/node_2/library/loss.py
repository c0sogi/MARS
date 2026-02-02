import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function is specific to the competition metric, which evaluates
    RMSE separately for specific columns and then averages them.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

        # Identify indices of columns that are scored
        self.target_cols = Config.TARGET_COLS
        self.scored_cols = Config.SCORED_COLS

        # Find indices: e.g., if targets are [A, B, C, D, E] and scored are [A, B, D]
        # indices would be [0, 1, 3]
        self.scored_indices = [
            i for i, col in enumerate(self.target_cols) if col in self.scored_cols
        ]

        # Register as buffer to ensure it moves to device with the module
        self.register_buffer(
            "scored_indices_tensor", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len_Pred, Num_Targets).
                                   Typically (B, 107, 5).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len_Target, Num_Targets).
                                    Typically (B, 68, 5).

        Returns:
            torch.Tensor: Scalar MCRMSE loss.
        """
        # 1. Slice inputs to match target sequence length
        # Predictions are usually for full length (107), but targets are only for first 68
        seq_len_target = targets.shape[1]
        inputs_sliced = inputs[:, :seq_len_target, :]

        # 2. Select only the scored columns
        # inputs_sliced: (B, 68, 5) -> (B, 68, 3)
        # targets: (B, 68, 5) -> (B, 68, 3)
        inputs_scored = torch.index_select(inputs_sliced, 2, self.scored_indices_tensor)
        targets_scored = torch.index_select(targets, 2, self.scored_indices_tensor)

        # 3. Calculate MSE per column
        # We average over Batch (dim 0) and Sequence (dim 1), keeping Columns (dim 2) separate initially
        mse_per_col = torch.mean((inputs_scored - targets_scored) ** 2, dim=(0, 1))

        # 4. Calculate RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # 5. Average RMSEs across columns to get MCRMSE
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse
