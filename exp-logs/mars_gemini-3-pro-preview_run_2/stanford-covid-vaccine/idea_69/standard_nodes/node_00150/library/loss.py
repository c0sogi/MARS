import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function strictly adheres to the competition scoring rules:
    1. It masks the sequence to only include the first 'seq_scored' positions (0-67).
    2. It filters the target columns to only include the scored conditions:
       - reactivity
       - deg_Mg_pH10
       - deg_Mg_50C
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.scored_len = Config.SCORED_LEN
        self.scored_indices = Config.SCORED_INDICES
        self.eps = 1e-6

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5).
            targets (torch.Tensor): Ground truth targets of shape (Batch, Seq_Len, 5).

        Returns:
            torch.Tensor: A scalar tensor representing the mean columnwise RMSE.
        """
        # 1. Slice Sequence Dimension
        # Keep only the valid scored positions (e.g., first 68 bases)
        # Shape changes from (B, 107, 5) -> (B, 68, 5)
        pred_scored = inputs[:, : self.scored_len, :]
        true_scored = targets[:, : self.scored_len, :]

        # 2. Slice Channel Dimension
        # Keep only the scored columns [0, 1, 3]
        # Shape changes from (B, 68, 5) -> (B, 68, 3)
        pred_scored = pred_scored[:, :, self.scored_indices]
        true_scored = true_scored[:, :, self.scored_indices]

        # 3. Calculate Squared Error
        squared_diff = (pred_scored - true_scored) ** 2

        # 4. Calculate MSE per column
        # Average over Batch (dim 0) and Sequence (dim 1)
        # Shape: (3,)
        mse_per_col = torch.mean(squared_diff, dim=(0, 1))

        # 5. Calculate RMSE per column
        # Add epsilon for numerical stability to avoid NaN gradients if MSE is 0
        rmse_per_col = torch.sqrt(mse_per_col + self.eps)

        # 6. Average across columns to get final MCRMSE
        loss = torch.mean(rmse_per_col)

        return loss
