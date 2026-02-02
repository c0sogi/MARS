import torch
import torch.nn as nn
from library.utils import mcrmse


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This module computes the MCRMSE loss specifically for the RNA degradation task.
    It handles:
    1. Truncating predictions and targets to the 'seq_scored' length (first 68 bases).
    2. Selecting only the scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C).
    3. Computing the column-wise RMSE and averaging them.
    """

    def __init__(self, scored_indices=None, seq_scored=68):
        """
        Args:
            scored_indices (list[int], optional): Indices of the columns to score.
                                                  Defaults to [0, 1, 3] corresponding to
                                                  ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'].
            seq_scored (int, optional): The number of sequence positions to score starting from index 0.
                                        Defaults to 68.
        """
        super().__init__()
        # Default indices correspond to: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
        # The full target list is usually: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
        if scored_indices is None:
            self.scored_indices = [0, 1, 3]
        else:
            self.scored_indices = scored_indices

        self.seq_scored = seq_scored

    def forward(self, inputs, targets):
        """
        Computes the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predictions from the model. Shape (Batch, SeqLen, Channels).
            targets (torch.Tensor): Ground truth targets. Shape (Batch, SeqLen, Channels).

        Returns:
            torch.Tensor: The scalar loss value.
        """
        # 1. Slice to the scored sequence length
        # The competition only scores the first 'seq_scored' positions (typically 68).
        # We slice both inputs and targets to ensure we don't compute loss on unscored regions.
        # Use min to prevent index errors if for some reason seq_len < seq_scored (unlikely here).
        slice_len = min(inputs.shape[1], self.seq_scored)

        inputs_sliced = inputs[:, :slice_len, :]
        targets_sliced = targets[:, :slice_len, :]

        # 2. Compute MCRMSE using the utility function
        # The utility function handles:
        # - Flattening the batch and sequence dimensions
        # - Filtering for the specific scored_indices
        # - Calculating RMSE per column and then the mean
        loss = mcrmse(targets_sliced, inputs_sliced, scored_indices=self.scored_indices)

        return loss
