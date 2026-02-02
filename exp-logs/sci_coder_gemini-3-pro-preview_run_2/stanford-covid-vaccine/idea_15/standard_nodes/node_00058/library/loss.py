import torch
import torch.nn as nn
from library.config import Config


class MaskedMCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE)
    specifically for the scored columns defined in the competition metric.

    This loss function masks out auxiliary targets (deg_pH10, deg_50C)
    to prevent negative transfer during optimization.
    """

    def __init__(self):
        super(MaskedMCRMSELoss, self).__init__()

        # Identify indices of the columns that count towards the score
        # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
        # SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
        # Expected indices: [0, 1, 3]
        target_cols = Config.TARGET_COLS
        scored_cols = Config.SCORED_COLS

        indices = [target_cols.index(col) for col in scored_cols]

        # Register indices as a buffer so they are saved with the state_dict
        # and moved to the correct device automatically.
        self.register_buffer("scored_indices", torch.tensor(indices, dtype=torch.long))

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predicted tensor of shape (Batch, SeqLen, 5) or (N, 5)
            targets: Ground truth tensor of shape (Batch, SeqLen, 5) or (N, 5)

        Returns:
            loss: Scalar MCRMSE loss.
        """
        # Select only the scored columns
        # Using index_select or slicing with the tensor buffer
        inputs_scored = inputs[..., self.scored_indices]
        targets_scored = targets[..., self.scored_indices]

        # Calculate Squared Error: (y_pred - y_true)^2
        squared_diff = (inputs_scored - targets_scored) ** 2

        # Calculate Mean Squared Error (MSE) per column
        # We average over all dimensions except the last one (columns)
        # If shape is (Batch, SeqLen, 3), we average over dim 0 and 1.
        # If shape is (N, 3), we average over dim 0.
        dims_to_reduce = list(range(squared_diff.ndim - 1))
        mse_per_col = torch.mean(squared_diff, dim=dims_to_reduce)

        # Calculate RMSE per column
        # Adding a small epsilon for numerical stability in sqrt, though usually
        # not strictly necessary if errors are non-zero.
        rmse_per_col = torch.sqrt(mse_per_col + 1e-8)

        # Calculate Mean Columnwise RMSE (average of the column RMSEs)
        loss = torch.mean(rmse_per_col)

        return loss
