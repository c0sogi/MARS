import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    As per the strategy (High-Capacity Stabilized Synthesis), this loss is calculated
    on ALL 5 target columns during training to enable Multi-Task Learning and
    regularization via auxiliary signals (deg_pH10, deg_50C), even though only
    3 are used for the final competition scoring.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.seq_scored = Config.SEQ_SCORED

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predictions from the model.
                                   Shape: (Batch, Seq_Len, Num_Targets)
                                   Typically (B, 107, 5).
            targets (torch.Tensor): Ground truth values.
                                    Shape: (Batch, Seq_Scored, Num_Targets)
                                    Typically (B, 68, 5).

        Returns:
            torch.Tensor: Scalar loss value representing the mean of column-wise RMSEs.
        """
        # Slice the predictions to match the length of the ground truth data
        # The model outputs predictions for the full sequence (107), but targets
        # are only provided for the first 68 bases.
        inputs_sliced = inputs[:, : self.seq_scored, :]

        # Ensure targets are the same size (handling potential edge cases in batching)
        # though typically targets are already (B, 68, 5)
        if targets.shape[1] != self.seq_scored:
            targets = targets[:, : self.seq_scored, :]

        # Calculate Squared Error: (inputs - targets)^2
        squared_diff = (inputs_sliced - targets) ** 2

        # Calculate Mean Squared Error (MSE) for each column separately.
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions.
        # Result shape: (Num_Targets,) -> (5,)
        column_mse = torch.mean(squared_diff, dim=(0, 1))

        # Calculate Root Mean Squared Error (RMSE) for each column
        column_rmse = torch.sqrt(column_mse)

        # Calculate the Mean of the RMSEs across all columns (MCRMSE)
        loss = torch.mean(column_rmse)

        return loss
