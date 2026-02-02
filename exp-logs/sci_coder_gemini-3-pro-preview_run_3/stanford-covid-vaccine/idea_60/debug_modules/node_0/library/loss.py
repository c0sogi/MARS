import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.

    This loss function is designed for the RNA degradation prediction task.
    It computes the RMSE for each of the 5 target columns separately and then
    averages them. This supports the Multi-Task Learning strategy by optimizing
    all available experimental conditions.

    Logic:
    1. Slice predictions and targets to the scored sequence length (first 68 bases).
    2. Compute MSE for each column (averaging over batch and sequence dimensions).
    3. Compute RMSE for each column.
    4. Return the mean of the column RMSEs.
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.seq_scored = Config.SEQ_SCORED

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Model predictions. Shape (Batch, Seq_Len, Num_Targets).
                                   Usually (B, 107, 5).
            targets (torch.Tensor): Ground truth values. Shape (Batch, Seq_Len, Num_Targets)
                                    or (Batch, Scored_Len, Num_Targets).

        Returns:
            torch.Tensor: Scalar MCRMSE loss.
        """
        # 1. Slice Sequence Dimension
        # We only calculate loss on the first 68 positions where ground truth exists.

        # Ensure inputs are sliced
        if inputs.shape[1] >= self.seq_scored:
            inputs_sliced = inputs[:, : self.seq_scored, :]
        else:
            # Fallback if input is already shorter than expected (unlikely in this pipeline)
            inputs_sliced = inputs

        # Ensure targets are sliced
        # Targets might be loaded as 107 (padded) or 68.
        if targets.shape[1] >= self.seq_scored:
            targets_sliced = targets[:, : self.seq_scored, :]
        else:
            targets_sliced = targets

        # 2. Compute Metric
        # Calculate Squared Error: (y_pred - y_true)^2
        squared_diff = (inputs_sliced - targets_sliced) ** 2

        # Calculate MSE per column: Mean over Batch (dim 0) and Sequence (dim 1)
        # Result shape: (Num_Targets,) i.e., (5,)
        mse_per_col = torch.mean(squared_diff, dim=(0, 1))

        # Calculate RMSE per column
        rmse_per_col = torch.sqrt(mse_per_col)

        # 3. Aggregate
        # MCRMSE is the mean of the RMSEs across columns
        loss = torch.mean(rmse_per_col)

        return loss
