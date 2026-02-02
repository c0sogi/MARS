import torch
import torch.nn as nn


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) for the
    competition-scored columns: reactivity, deg_Mg_pH10, and deg_Mg_50C.

    This loss is designed to handle the specific masking requirements of the RNA
    degradation task, where only the first 'seq_scored' positions are valid.
    """

    def __init__(self):
        super().__init__()
        # Indices corresponding to ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
        # within the 5-channel target vector.
        # 0: reactivity
        # 1: deg_Mg_pH10
        # 3: deg_Mg_50C
        self.scored_indices = [0, 1, 3]

    def forward(self, inputs, targets, mask=None):
        """
        Compute the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, 5).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, 5).
            mask (torch.Tensor, optional): Boolean or Float mask of shape (Batch, Seq_Len).
                                           Values should be 1 for valid positions, 0 for padding/unscored.

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # Slice the inputs and targets to keep only the scored columns
        # Robust check: if inputs already have 3 channels, assume they are the scored ones.
        if inputs.shape[-1] == 5:
            inputs_scored = inputs[:, :, self.scored_indices]
        else:
            inputs_scored = inputs

        if targets.shape[-1] == 5:
            targets_scored = targets[:, :, self.scored_indices]
        else:
            targets_scored = targets

        # Calculate Squared Errors: (y_hat - y)^2
        squared_diff = (inputs_scored - targets_scored) ** 2

        if mask is not None:
            # Ensure mask is float and has correct dimensions for broadcasting
            # mask: (B, L) -> (B, L, 1)
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = mask.float()

            # Apply mask to zero out errors at invalid positions
            squared_diff = squared_diff * mask

            # Calculate the number of valid elements
            # We sum the mask over Batch and Sequence dimensions.
            # Since the mask is shared across the 3 columns, the count is the same for each column.
            # valid_count shape: scalar
            valid_count = mask.sum()

            # Avoid division by zero
            valid_count = torch.clamp(valid_count, min=1.0)

            # Compute MSE for each column: Sum(Squared Errors) / Count
            # Sum over Batch (0) and Sequence (1) dimensions, keeping Channel (2)
            mse_per_col = squared_diff.sum(dim=(0, 1)) / valid_count

        else:
            # If no mask is provided, compute standard mean over batch and sequence
            mse_per_col = squared_diff.mean(dim=(0, 1))

        # Compute RMSE for each column
        # Add a small epsilon to ensure numerical stability during backprop
        rmse_per_col = torch.sqrt(mse_per_col + 1e-8)

        # Compute the mean of the RMSEs (MCRMSE)
        loss = torch.mean(rmse_per_col)

        return loss
