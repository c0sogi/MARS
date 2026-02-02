import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function is designed to optimize all 5 target columns provided in the
    training data (reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C)
    without inverse-variance weighting, as per the 'Deep Bias-Refined Decoupled
    Post-Norm BiGRU' strategy.

    It explicitly handles the sequence length constraint by slicing predictions
    and targets to the scored sequence length (Config.PRED_LEN = 68) before
    computation.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Forward pass for the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predictions from the model.
                                   Shape: (Batch, Seq_Len, 5)
            targets (torch.Tensor): Ground truth values.
                                    Shape: (Batch, Seq_Len, 5)

        Returns:
            torch.Tensor: Scalar loss value representing the mean of the RMSEs
                          across the 5 target columns.
        """
        # 1. Slice to the scored sequence length (first 68 bases)
        # The experimental data is only valid for the first 68 positions.
        # We ignore the predictions/targets for positions > 67.
        pred_scored = inputs[:, : Config.PRED_LEN, :]
        true_scored = targets[:, : Config.PRED_LEN, :]

        # 2. Calculate MSE per column
        # We compute the mean squared error over the batch (dim 0) and sequence (dim 1)
        # resulting in a tensor of shape (5,) representing MSE for each target type.
        mse_per_col = torch.mean((pred_scored - true_scored) ** 2, dim=(0, 1))

        # 3. Calculate RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # 4. Average the RMSEs across columns to get the final scalar loss
        loss = torch.mean(rmse_per_col)

        return loss
