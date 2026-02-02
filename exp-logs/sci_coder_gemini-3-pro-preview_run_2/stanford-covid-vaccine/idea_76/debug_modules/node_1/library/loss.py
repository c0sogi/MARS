import torch
import torch.nn as nn
from library.config import Config


class AnchoredMCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) with Boundary Anchoring.

    This loss function is critical for the AHC-HIDN strategy. It:
    1. Filters predictions and targets to only the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    2. Computes the error over the full sequence length (0-107) rather than just the scored length.
       This enforces 'Boundary Anchoring', compelling the model to predict the neutral baseline (0.0)
       in the tail regions, which stabilizes the Bidirectional RNN's hidden states.
    """

    def __init__(self):
        super().__init__()
        # Load indices for reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
        self.scored_indices = Config.SCORED_COLS_INDICES

    def forward(self, preds, targets):
        """
        Computes the Anchored MCRMSE loss.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Length, 5).
            targets (torch.Tensor): Ground truth of shape (Batch, Length, 5).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Column Filtering
        # Select only the columns relevant to the competition score.
        # preds/targets become shape (Batch, Length, 3)
        p = preds[:, :, self.scored_indices]
        t = targets[:, :, self.scored_indices]

        # 2. Squared Error Calculation
        # Compute element-wise squared error
        squared_error = (p - t) ** 2

        # 3. Anchored MSE Calculation
        # Calculate Mean Squared Error per column.
        # Crucially, we average over the Batch (dim 0) and the FULL Sequence Length (dim 1).
        # We do NOT slice to Config.SEQ_SCORED here. Including the tail (68-107) in the
        # mean calculation implements the Boundary Anchoring strategy.
        mse_per_column = torch.mean(squared_error, dim=(0, 1))

        # 4. RMSE Calculation
        # Take the square root to get RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 5. Final Aggregation
        # Average the RMSEs across the 3 columns to get MCRMSE
        loss = torch.mean(rmse_per_column)

        return loss
