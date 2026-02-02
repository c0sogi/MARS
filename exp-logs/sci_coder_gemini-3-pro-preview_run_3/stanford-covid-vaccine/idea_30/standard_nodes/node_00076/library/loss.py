import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function computes the RMSE for each of the 5 target columns separately
    and then takes the average of these RMSEs. It explicitly handles the slicing
    of sequences to the scored length (Config.PRED_LEN) to ensure the loss is
    calculated only on the valid ground truth positions.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        # Small epsilon to prevent NaN gradients when MSE is 0 (derivative of sqrt(0) is undefined)
        self.eps = 1e-8

    def forward(self, inputs, targets):
        """
        Compute the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predictions from the model.
                                   Shape: (Batch, Seq_Len, Num_Targets)
            targets (torch.Tensor): Ground truth values.
                                    Shape: (Batch, Seq_Len, Num_Targets)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Slice inputs and targets to the scored length (first 68 positions)
        # as specified in the competition rules and Config.
        # inputs shape becomes: (Batch, 68, 5)
        preds_sliced = inputs[:, : Config.PRED_LEN, :]
        targets_sliced = targets[:, : Config.PRED_LEN, :]

        # Calculate Squared Error: (y - y_hat)^2
        squared_diff = (preds_sliced - targets_sliced) ** 2

        # Calculate Mean Squared Error (MSE) for each column.
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions.
        # Result shape: (Num_Targets,) -> (5,)
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # Calculate Root Mean Squared Error (RMSE) for each column.
        # Add epsilon for numerical stability.
        rmse_per_column = torch.sqrt(mse_per_column + self.eps)

        # Calculate the Mean of the column RMSEs (MCRMSE).
        # Result is a scalar.
        loss = torch.mean(rmse_per_column)

        return loss
