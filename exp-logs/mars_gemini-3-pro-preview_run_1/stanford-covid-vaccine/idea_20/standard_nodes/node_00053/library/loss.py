import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Computes Mean Squared Error Loss only on masked positions.
    Used to train on the specific 'seq_scored' positions (first 68) where ground truth exists.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets, mask):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, n_targets).
            targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, n_targets).
            mask (torch.Tensor): Boolean mask of shape (Batch, Seq_Len). True indicates a valid scored position.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Select valid positions based on the mask
        # inputs[mask] flattens the batch and sequence dimensions for valid positions
        # Shape becomes (Total_Valid_Positions, n_targets)
        valid_inputs = inputs[mask]
        valid_targets = targets[mask]

        # Compute MSE on the valid data
        return self.mse(valid_inputs, valid_targets)


def calculate_mcrmse(inputs, targets, mask):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) on masked positions.

    Logic:
    1. Filter inputs and targets using the mask.
    2. Compute MSE for each target column independently.
    3. Compute RMSE for each column.
    4. Average the RMSE values across columns.

    Args:
        inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, n_targets).
        targets (torch.Tensor): Ground truth of shape (Batch, Seq_Len, n_targets).
        mask (torch.Tensor): Boolean mask of shape (Batch, Seq_Len).

    Returns:
        torch.Tensor: Scalar MCRMSE value.
    """
    with torch.no_grad():
        # Select valid positions based on the mask
        # Shape: (Total_Valid_Positions, n_targets)
        valid_inputs = inputs[mask]
        valid_targets = targets[mask]

        # Compute MSE per column (dim=0 is the flattened sample dimension)
        mse_per_col = torch.mean((valid_inputs - valid_targets) ** 2, dim=0)

        # Compute RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # Average RMSE across all columns
        mcrmse = torch.mean(rmse_per_col)

    return mcrmse
