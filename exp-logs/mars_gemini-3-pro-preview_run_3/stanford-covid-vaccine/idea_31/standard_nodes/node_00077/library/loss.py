import torch
import torch.nn as nn


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function calculates the RMSE for each target column separately
    and then takes the average of these RMSEs. It is the primary metric
    for the RNA degradation prediction task.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        # Epsilon to prevent NaN gradients during backprop if MSE is 0
        self.eps = 1e-6

    def forward(self, inputs, targets):
        """
        Computes the MCRMSE loss between predictions and targets.

        Args:
            inputs (torch.Tensor): Predicted tensor of shape (Batch, SeqLen, Channels)
                                   or (Batch, Channels).
            targets (torch.Tensor): Ground truth tensor of the same shape as inputs.

        Returns:
            torch.Tensor: A scalar tensor representing the mean columnwise RMSE.
        """
        # Ensure inputs and targets are compatible
        if inputs.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: inputs {inputs.shape} vs targets {targets.shape}"
            )

        # Calculate squared errors: (y - y_hat)^2
        squared_diff = (inputs - targets) ** 2

        # Identify dimensions to reduce.
        # We want to keep the last dimension (Channels/Columns) and average over the rest.
        # For shape (N, L, C), we reduce over (0, 1).
        # For shape (N, C), we reduce over (0).
        dims_to_reduce = list(range(inputs.ndim - 1))

        # Calculate Mean Squared Error (MSE) per column
        mse = torch.mean(squared_diff, dim=dims_to_reduce)

        # Calculate Root Mean Squared Error (RMSE) per column
        # Add epsilon for numerical stability
        rmse = torch.sqrt(mse + self.eps)

        # Calculate the mean of the column RMSEs
        mcrmse = torch.mean(rmse)

        return mcrmse
