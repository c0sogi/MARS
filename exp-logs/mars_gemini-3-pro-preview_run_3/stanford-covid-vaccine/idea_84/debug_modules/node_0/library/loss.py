import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    Computes the average RMSE across specified target columns.
    Automatically slices the input sequences to the scored length (Config.SEQ_SCORED).
    """

    def __init__(self, seq_scored=Config.SEQ_SCORED):
        """
        Args:
            seq_scored (int): The number of positions from the start of the sequence
                              to include in the loss calculation. Defaults to Config.SEQ_SCORED (68).
        """
        super().__init__()
        self.seq_scored = seq_scored

    def forward(self, inputs, targets, column_indices=None):
        """
        Calculate MCRMSE.

        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, SeqLen, NumTargets).
            targets (torch.Tensor): Ground truth of shape (Batch, SeqLen, NumTargets).
            column_indices (list[int] or torch.Tensor, optional): Indices of columns to include
                                                                  in the calculation. If None, uses all columns.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Slice to scored sequence length
        # The competition only scores the first 68 bases.
        # Inputs/Targets shape becomes: (Batch, seq_scored, NumTargets)
        inputs_sliced = inputs[:, : self.seq_scored, :]
        targets_sliced = targets[:, : self.seq_scored, :]

        # 2. Filter columns if specified
        # Used to distinguish between training on all 5 targets vs validation on 3 scored targets.
        if column_indices is not None:
            inputs_sliced = inputs_sliced[:, :, column_indices]
            targets_sliced = targets_sliced[:, :, column_indices]

        # 3. Calculate MSE per column
        # We average over Batch (dim 0) and Sequence (dim 1), keeping Targets (dim 2) separate.
        # Shape: (NumSelectedTargets,)
        diff = inputs_sliced - targets_sliced
        mse_per_column = torch.mean(diff**2, dim=(0, 1))

        # 4. Calculate RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # 5. Average RMSE across columns (MCRMSE)
        mcrmse = torch.mean(rmse_per_column)

        return mcrmse
