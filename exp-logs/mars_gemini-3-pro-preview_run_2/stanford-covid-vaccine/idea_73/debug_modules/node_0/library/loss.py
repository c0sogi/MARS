import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This loss function is configured to support the 'Boundary Anchoring' strategy.
    When enabled, the loss is computed over the full sequence length (0-107) rather
    than just the scored positions (0-68). This anchors the hidden states of the
    RNN in the tail region, preventing noise propagation back into the valid region.
    """

    def __init__(self):
        super().__init__()
        # Indices of the columns to be scored: [0, 1, 3]
        # (reactivity, deg_Mg_pH10, deg_Mg_50C)
        self.scored_indices = Config.SCORED_INDICES

        # Configuration for sequence length handling
        self.boundary_anchoring = Config.BOUNDARY_ANCHORING
        self.scorable_length = Config.SCORABLE_LENGTH

    def forward(self, inputs, targets):
        """
        Computes the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predicted values. Shape (Batch, Length, 5).
            targets (torch.Tensor): Ground truth values. Shape (Batch, Length, 5).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Select Scored Columns
        # We filter the inputs and targets to only include the columns that are
        # part of the competition metric.
        inputs_scored = inputs[:, :, self.scored_indices]
        targets_scored = targets[:, :, self.scored_indices]

        # 2. Apply Sequence Length Masking (if necessary)
        # If Boundary Anchoring is disabled, we strictly evaluate on the first 68 positions.
        # If enabled (default for this Idea), we evaluate on the full length (107).
        # The dataset provides 0.0 targets for positions 68-107, so the model learns
        # to predict 0.0 in the tail.
        if not self.boundary_anchoring:
            inputs_scored = inputs_scored[:, : self.scorable_length, :]
            targets_scored = targets_scored[:, : self.scorable_length, :]

        # 3. Compute MSE per Column
        # We calculate the mean squared error across the Batch (dim 0) and Sequence (dim 1)
        # dimensions independently for each column.
        mse = torch.mean((inputs_scored - targets_scored) ** 2, dim=(0, 1))

        # 4. Compute RMSE per Column
        rmse = torch.sqrt(mse)

        # 5. Aggregate to MCRMSE
        # Average the RMSE values across the columns.
        loss = torch.mean(rmse)

        return loss
