import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function strictly adheres to the competition scoring rules:
    1. Only the first 'seq_scored' positions (typically 68) are considered.
    2. Only specific columns (reactivity, deg_Mg_pH10, deg_Mg_50C) are scored.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

        # Retrieve column definitions from Config
        self.target_cols = Config.TARGET_COLS
        self.scored_cols = Config.SCORED_COLS
        self.seq_scored = Config.SEQ_SCORED

        # Determine the indices of the columns that contribute to the score.
        # Based on standard config:
        # Targets: ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # Scored:  ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        # Indices: [0, 1, 3]
        self.scored_indices = [
            i for i, col in enumerate(self.target_cols) if col in self.scored_cols
        ]

        # Register indices as a buffer to ensure they move to the correct device with the module
        self.register_buffer(
            "scored_indices_tensor", torch.tensor(self.scored_indices, dtype=torch.long)
        )

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Targets).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, Num_Targets).

        Returns:
            torch.Tensor: Scalar MCRMSE loss.
        """
        # 1. Slice Sequence Length
        # We only evaluate the first 'seq_scored' positions (0-67).
        # Inputs/Targets outside this range are ignored.
        pred_scored = inputs[:, : self.seq_scored, :]
        true_scored = targets[:, : self.seq_scored, :]

        # 2. Select Scored Columns
        # Filter down to only the indices corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C.
        # We use the registered buffer for device safety.
        pred_scored = torch.index_select(pred_scored, 2, self.scored_indices_tensor)
        true_scored = torch.index_select(true_scored, 2, self.scored_indices_tensor)

        # 3. Compute MSE per column
        # We calculate the mean squared error over the Batch (dim 0) and Sequence (dim 1) dimensions.
        # Result shape: (Num_Scored_Cols,) i.e., (3,)
        mse = torch.mean((pred_scored - true_scored) ** 2, dim=(0, 1))

        # 4. Compute RMSE per column
        rmse = torch.sqrt(mse)

        # 5. Average RMSEs to get MCRMSE
        loss = torch.mean(rmse)

        return loss
