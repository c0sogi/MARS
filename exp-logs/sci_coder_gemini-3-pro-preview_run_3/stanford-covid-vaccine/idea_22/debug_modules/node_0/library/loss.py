import torch
import torch.nn as nn


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function computes the Root Mean Squared Error (RMSE) for each
    target column independently and then averages them. This aligns with the
    competition metric and the training strategy defined in Idea 22, which
    requires optimizing all 5 target columns (reactivity, deg_Mg_pH10,
    deg_pH10, deg_Mg_50C, deg_50C) without inverse-variance weighting.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Compute the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predicted values.
                                   Shape: (Batch_Size, Seq_Len, Num_Columns) or (N, Num_Columns).
            targets (torch.Tensor): Ground truth values.
                                    Shape: Same as inputs.

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # Ensure inputs and targets share the same shape
        assert (
            inputs.shape == targets.shape
        ), f"Shape mismatch: {inputs.shape} vs {targets.shape}"

        # Determine the number of columns (last dimension)
        num_columns = inputs.shape[-1]

        # Flatten the batch and sequence dimensions to (N_samples, Num_Columns)
        # This handles both (Batch, Seq, Cols) and (N, Cols) formats uniformly
        inputs_flat = inputs.view(-1, num_columns)
        targets_flat = targets.view(-1, num_columns)

        # Calculate Mean Squared Error (MSE) for each column independently
        # dim=0 aggregates over the flattened samples
        col_mse = torch.mean((inputs_flat - targets_flat) ** 2, dim=0)

        # Calculate Root Mean Squared Error (RMSE) for each column
        # Add a small epsilon to avoid NaN gradients if MSE is exactly 0
        col_rmse = torch.sqrt(col_mse + 1e-8)

        # Calculate the mean of the column RMSEs (MCRMSE)
        loss = torch.mean(col_rmse)

        return loss
