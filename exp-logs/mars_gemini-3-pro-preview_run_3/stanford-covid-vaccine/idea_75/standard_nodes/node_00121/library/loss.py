import torch
import torch.nn as nn


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    Calculates the MCRMSE between predictions and targets.
    The metric is computed as the mean of the RMSEs for each target column.

    Formula:
        MCRMSE = (1/M) * sum_{j=1}^{M} sqrt( (1/N) * sum_{i=1}^{N} (y_{ij} - y_hat_{ij})^2 )

    Where:
        M = number of target columns (5)
        N = total number of scored positions (Batch Size * seq_scored)
    """

    def __init__(self, seq_scored: int = 68):
        """
        Initialize the MCRMSELoss module.

        Args:
            seq_scored (int): The number of sequence positions to include in the loss calculation.
                              Defaults to 68 based on the dataset specifications.
        """
        super(MCRMSELoss, self).__init__()
        self.seq_scored = seq_scored

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predicted values of shape (Batch, Seq_Len, Num_Targets).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Num_Targets).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Slice inputs and targets to the scored sequence length
        # We assume the sequence dimension is dim=1
        preds_sliced = inputs[:, : self.seq_scored, :]
        targs_sliced = targets[:, : self.seq_scored, :]

        # Calculate Squared Error: (y - y_hat)^2
        squared_diff = (preds_sliced - targs_sliced) ** 2

        # Calculate Mean Squared Error (MSE) for each column
        # We average over the batch (dim 0) and sequence (dim 1) dimensions
        # resulting in a tensor of shape (Num_Targets,)
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # Calculate Root Mean Squared Error (RMSE) for each column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Calculate the mean of the RMSEs across all columns
        loss = torch.mean(rmse_per_column)

        return loss
