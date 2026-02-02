import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Implements the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function is designed to support the 'Boundary Anchoring' strategy
    by calculating the error across the full sequence length (e.g., 107) provided
    in the input tensors, rather than masking specific regions.

    Formula:
        MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

        Where:
        - Nt is the number of target columns (channels).
        - n is the total number of elements per column (Batch Size * Sequence Length).
        - i indexes over all samples and sequence positions.
        - j indexes over the target columns.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, preds, targets):
        """
        Calculates the MCRMSE loss.

        Args:
            preds (torch.Tensor): Predicted values of shape (Batch, Seq_Len, Channels).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Channels).
                                    For Boundary Anchoring, the tail regions of targets
                                    should be filled with 0.0 (or appropriate baseline).

        Returns:
            torch.Tensor: A scalar tensor representing the mean columnwise RMSE.
        """
        # Calculate Squared Error: (B, L, C)
        squared_error = (preds - targets) ** 2

        # Calculate Mean Squared Error (MSE) per column: (C,)
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions.
        # This includes the 'anchoring' tail positions in the calculation.
        mse_per_column = torch.mean(squared_error, dim=(0, 1))

        # Calculate Root Mean Squared Error (RMSE) per column: (C,)
        rmse_per_column = torch.sqrt(mse_per_column)

        # Calculate the mean across all columns to get MCRMSE: Scalar
        mcrmse = torch.mean(rmse_per_column)

        return mcrmse
