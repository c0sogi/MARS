import torch
import torch.nn as nn
from library.config import SCORED_LENGTH, TARGET_COLS, SCORED_COLS


class MaskedMCRMSE(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) with strict sequence masking.

    This loss function:
    1. Masks the sequence to the first `SCORED_LENGTH` positions (typically 0-67).
    2. Selects only the columns specified in `SCORED_COLS`.
    3. Computes the RMSE for each selected column independently.
    4. Returns the average of these RMSE values.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

        # Determine indices of columns to score based on config
        # TARGET_COLS: ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # SCORED_COLS: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        # Resulting indices: [0, 1, 3]
        self.scored_indices = [
            i for i, col in enumerate(TARGET_COLS) if col in SCORED_COLS
        ]

        # Register as buffer to ensure it moves to device with the module
        self.register_buffer(
            "scored_indices_tensor", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Predictions tensor of shape (Batch, Length, Channels)
            targets: Ground truth tensor of shape (Batch, Length, Channels)

        Returns:
            mcrmse: Scalar tensor representing the loss
        """
        # 1. Strict Sequence Masking
        # We only care about the first SCORED_LENGTH positions (0-67)
        # Slicing assumes the length dimension is dim 1: (N, L, C)
        preds_masked = preds[:, :SCORED_LENGTH, :]
        targets_masked = targets[:, :SCORED_LENGTH, :]

        # 2. Column Selection
        # Select only the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
        # We use index_select along the channel dimension (dim 2)
        preds_selected = torch.index_select(preds_masked, 2, self.scored_indices_tensor)
        targets_selected = torch.index_select(
            targets_masked, 2, self.scored_indices_tensor
        )

        # 3. Compute Squared Errors
        mse = (preds_selected - targets_selected) ** 2

        # 4. Compute Mean Squared Error per column
        # Average over Batch (dim 0) and Sequence Length (dim 1) to get MSE per channel
        mse_per_col = torch.mean(mse, dim=(0, 1))

        # 5. Compute RMSE per column
        # Add epsilon for numerical stability during backprop (gradient of sqrt(0) is undefined)
        rmse_per_col = torch.sqrt(mse_per_col + self.eps)

        # 6. Compute Mean of RMSEs (MCRMSE)
        # Average the RMSE values across the selected columns
        mcrmse = torch.mean(rmse_per_col)

        return mcrmse
