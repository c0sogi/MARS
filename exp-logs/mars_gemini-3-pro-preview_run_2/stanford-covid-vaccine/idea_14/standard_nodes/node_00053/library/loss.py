import torch
import torch.nn as nn
from library.config import mcrmse_loss


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function is designed to optimize the model based on the specific
    columns scored in the competition:
    - reactivity (Index 0)
    - deg_Mg_pH10 (Index 1)
    - deg_Mg_50C (Index 3)

    It ignores auxiliary columns (deg_pH10, deg_50C) to prevent negative transfer
    and align the training objective with the evaluation metric.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Calculates the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predicted tensor of shape (Batch, Length, 5).
            targets (torch.Tensor): Ground truth tensor of shape (Batch, Length, 5).

        Returns:
            torch.Tensor: Scalar loss value representing the mean RMSE across the scored columns.
        """
        # Delegate calculation to the pre-defined function in library.config
        # which handles column slicing and RMSE aggregation.
        return mcrmse_loss(inputs, targets)
