import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Computes the Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function is designed for the RNA degradation task where predictions
    are made for the full sequence length (107), but ground truth is only available
    and scored for the first 68 positions.

    Formula:
    MCRMSE = (1/Nt) * Sum_j( sqrt( (1/n) * Sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Where:
    - Nt is the number of target columns.
    - n is the total number of scored positions (Batch Size * Seq_Scored).
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.seq_scored = Config.SEQ_SCORED

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted values.
                                   Shape: (Batch, Seq_Len, Num_Targets)
                                   Typically (B, 107, 5).
            targets (torch.Tensor): Ground truth values.
                                    Shape: (Batch, Seq_Scored, Num_Targets)
                                    Typically (B, 68, 5).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Slice the inputs to match the scored sequence length (first 68 positions)
        # inputs shape becomes: (Batch, 68, Num_Targets)
        pred_scored = inputs[:, : self.seq_scored, :]

        # Ensure targets are also sliced to the scored length if they happen to be longer
        # (though typically data loaders provide them as 68)
        true_scored = targets[:, : self.seq_scored, :]

        # Calculate Squared Error
        squared_diff = (pred_scored - true_scored) ** 2

        # Calculate Mean Squared Error (MSE) per column
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions simultaneously.
        # This corresponds to the (1/n) term inside the square root in the formula.
        # Result shape: (Num_Targets,)
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # Calculate Root Mean Squared Error (RMSE) per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Calculate the Mean of the RMSEs across all columns
        # This corresponds to the (1/Nt) summation term.
        loss = torch.mean(rmse_per_column)

        return loss
