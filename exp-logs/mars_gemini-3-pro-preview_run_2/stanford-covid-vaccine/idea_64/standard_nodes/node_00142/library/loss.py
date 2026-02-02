import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) strictly on
    valid scored positions and scored columns.

    This loss function:
    1. Filters inputs/targets to only include scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    2. Applies a boolean mask to exclude zero-padded positions (indices > 67).
    3. Computes RMSE per column and averages them.
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()

        # Determine indices of the columns that are actually scored
        # Config.TARGET_COLS contains all predicted columns (5)
        # Config.SCORED_COLS contains the subset used for the metric (3)
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]

        # Register indices as a buffer so it moves to the correct device with the module
        self.register_buffer(
            "scored_idxs", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, inputs, targets, mask=None):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, SeqLen, NumTargets).
            targets (torch.Tensor): Ground Truth of shape (Batch, SeqLen, NumTargets).
            mask (torch.Tensor, optional): Boolean or Binary mask of shape (Batch, SeqLen)
                                           indicating valid positions (e.g., indices 0-67).
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Select only the scored columns from the channel dimension (last dim)
        # inputs shape: (B, L, 5) -> (B, L, 3)
        inputs_scored = inputs[..., self.scored_idxs]
        targets_scored = targets[..., self.scored_idxs]

        # 2. Apply Masking
        if mask is not None:
            # Ensure mask is boolean
            mask_bool = mask > 0

            # Advanced Indexing:
            # Using a boolean mask of shape (B, L) on a tensor of shape (B, L, C)
            # returns a tensor of shape (N_valid, C), where N_valid is the number of True elements.
            # This effectively flattens the batch and sequence dimensions for valid positions
            # while preserving the channel separation needed for column-wise RMSE.
            inputs_flat = inputs_scored[mask_bool]
            targets_flat = targets_scored[mask_bool]
        else:
            # If no mask is provided, flatten batch and sequence dimensions entirely
            inputs_flat = inputs_scored.reshape(-1, len(self.scored_indices))
            targets_flat = targets_scored.reshape(-1, len(self.scored_indices))

        # 3. Compute Column-wise MSE
        # Calculate squared difference
        squared_diff = (inputs_flat - targets_flat) ** 2

        # Mean over the valid samples dimension (dim=0), keeping channel dimension
        mse_per_col = torch.mean(squared_diff, dim=0)

        # 4. Compute RMSE per column
        # Add a small epsilon for numerical stability to avoid NaN gradients at 0
        rmse_per_col = torch.sqrt(mse_per_col + 1e-16)

        # 5. Average RMSEs across columns to get the final MCRMSE
        loss = torch.mean(rmse_per_col)

        return loss
