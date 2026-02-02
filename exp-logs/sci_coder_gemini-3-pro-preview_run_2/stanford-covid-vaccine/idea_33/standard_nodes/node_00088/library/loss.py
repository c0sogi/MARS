import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function is specific to the competition metric. It:
    1. Restricts calculation to the first `Config.SCORED_SEQ_LENGTH` positions (68).
    2. Selects only the columns specified in `Config.SCORED_COLS` for scoring.
    3. Computes RMSE for each selected column independently and then averages them.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

        # Identify indices of the columns that contribute to the score
        # Config.TARGET_COLS: ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Config.SCORED_COLS: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        self.scored_indices = [
            i for i, col in enumerate(Config.TARGET_COLS) if col in Config.SCORED_COLS
        ]

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Predicted values of shape (Batch, Seq_Len, Output_Dim).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Output_Dim).

        Returns:
            torch.Tensor: Scalar MCRMSE loss.
        """
        # 1. Slice Sequence Dimension
        # Only the first 68 bases are scored.
        # Shape becomes: (Batch, 68, Output_Dim)
        pred_scored = inputs[:, : Config.SCORED_SEQ_LENGTH, :]
        true_scored = targets[:, : Config.SCORED_SEQ_LENGTH, :]

        # 2. Select Scored Columns
        # Filter to keep only reactivity, deg_Mg_pH10, and deg_Mg_50C.
        # Shape becomes: (Batch, 68, 3)
        # We use the pre-calculated indices.
        pred_scored = pred_scored[:, :, self.scored_indices]
        true_scored = true_scored[:, :, self.scored_indices]

        # 3. Compute MSE per Column
        # We calculate the mean squared error across the Batch and Sequence dimensions (dim 0 and 1).
        # Result shape: (3,) -> one MSE value per scored column.
        mse_per_column = torch.mean((true_scored - pred_scored) ** 2, dim=(0, 1))

        # 4. Compute RMSE per Column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 5. Compute Mean of RMSEs (MCRMSE)
        loss = torch.mean(rmse_per_column)

        return loss
