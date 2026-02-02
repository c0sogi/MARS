import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    MCRMSELoss implements the Mean Columnwise Root Mean Squared Error loss function.

    This loss is designed for the Multi-Task Learning strategy, optimizing
    all 5 target columns (reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C)
    simultaneously. It handles the sequence length discrepancy between predictions
    and targets by slicing the predictions.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Compute the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Model predictions of shape (Batch, Seq_Len_Out, Num_Targets).
                                   Typically (Batch, 107, 5).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len_Scored, Num_Targets).
                                    Typically (Batch, 68, 5).

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # Determine the scored sequence length from the targets (usually 68)
        # This ensures we only evaluate positions for which we have ground truth.
        scored_len = targets.shape[1]

        # Slice the inputs to match the target sequence length
        # inputs: (B, 107, 5) -> (B, 68, 5)
        inputs_sliced = inputs[:, :scored_len, :]

        # Compute squared differences
        # Shape: (B, 68, 5)
        diff = inputs_sliced - targets
        loss_sq = diff**2

        # Compute Mean Squared Error (MSE) per column
        # Average over Batch (dim 0) and Sequence (dim 1) to get one MSE value per target type
        # Shape: (5,)
        mse_per_column = torch.mean(loss_sq, dim=(0, 1))

        # Compute Root Mean Squared Error (RMSE) per column
        # Add a small epsilon for numerical stability to prevent NaN gradients if MSE is exactly 0
        rmse_per_column = torch.sqrt(mse_per_column + 1e-6)

        # Compute the mean of RMSEs across all 5 columns to get the final scalar loss
        loss = torch.mean(rmse_per_column)

        return loss
