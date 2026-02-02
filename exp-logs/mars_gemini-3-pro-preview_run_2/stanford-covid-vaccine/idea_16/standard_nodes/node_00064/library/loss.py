import torch
import torch.nn as nn
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This loss function specifically masks the calculation to only consider
    the scored columns defined in the configuration (reactivity, deg_Mg_pH10, deg_Mg_50C)
    to align the optimization objective with the competition metric and prevent
    negative transfer from auxiliary targets.
    """

    def __init__(self, scored_indices=None):
        """
        Args:
            scored_indices (list of int, optional): Indices of the target columns to include
                                                    in the loss calculation. Defaults to
                                                    Config.SCORED_TARGET_INDICES.
        """
        super(MCRMSELoss, self).__init__()
        if scored_indices is None:
            self.scored_indices = Config.SCORED_TARGET_INDICES
        else:
            self.scored_indices = scored_indices

    def forward(self, preds, targets):
        """
        Computes the MCRMSE loss.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, Num_Targets).
            targets (torch.Tensor): Ground truth targets of shape (Batch, Seq_Len, Num_Targets).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Select only the scored columns based on the indices
        # Assumes shape is (Batch, Seq_Len, Channels)
        preds_scored = preds[..., self.scored_indices]
        targets_scored = targets[..., self.scored_indices]

        # Calculate Mean Squared Error (MSE) for each column separately
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions
        mse = torch.mean((preds_scored - targets_scored) ** 2, dim=(0, 1))

        # Calculate Root Mean Squared Error (RMSE) per column
        rmse = torch.sqrt(mse)

        # Calculate the mean of the RMSEs across the columns
        loss = torch.mean(rmse)

        return loss
