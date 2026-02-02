import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Mean Columnwise Root Mean Squared Error (MCRMSE) Loss.

    This loss function calculates the RMSE for each of the 5 target columns separately
    and then takes the average. It automatically handles the slicing of the input
    sequence to match the length of the provided targets (seq_scored=68).
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.seq_scored = Config.SEQ_SCORED

    def forward(self, inputs, targets):
        """
        Calculate the MCRMSE loss.

        Args:
            inputs (torch.Tensor): Predicted values of shape (Batch, Seq_Len, 5).
                                   Seq_Len is typically 107 (full sequence).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Scored, 5).
                                    Seq_Scored is typically 68.

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # Slice inputs to match target length if necessary.
        # The model predicts for all 107 positions, but we only calculate loss
        # on the first 68 positions where ground truth exists.
        if inputs.shape[1] > targets.shape[1]:
            inputs = inputs[:, : targets.shape[1], :]

        # Calculate MSE for each target column separately.
        # We average over the Batch (dim 0) and Sequence (dim 1) dimensions.
        # Result shape: (5,)
        columnwise_mse = torch.mean((inputs - targets) ** 2, dim=(0, 1))

        # Calculate RMSE for each column
        columnwise_rmse = torch.sqrt(columnwise_mse)

        # Average the RMSEs across the 5 columns to get the final loss
        loss = torch.mean(columnwise_rmse)

        return loss


def mcrmse_metric(inputs, targets, scored_only=False):
    """
    Functional calculation of MCRMSE for validation and evaluation.

    Args:
        inputs (torch.Tensor or np.ndarray): Predictions.
        targets (torch.Tensor or np.ndarray): Ground truth.
        scored_only (bool): If True, calculates the metric only on the 3 columns
                            scored in the competition leaderboard:
                            [reactivity, deg_Mg_pH10, deg_Mg_50C].

    Returns:
        float: The calculated MCRMSE value.
    """
    # Ensure inputs are tensors
    if not torch.is_tensor(inputs):
        inputs = torch.tensor(inputs)
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)

    # Move to same device if necessary
    if inputs.device != targets.device:
        inputs = inputs.to(targets.device)

    # Slice inputs to match target length
    if inputs.shape[1] > targets.shape[1]:
        inputs = inputs[:, : targets.shape[1], :]

    # Calculate MSE per column
    columnwise_mse = torch.mean((inputs - targets) ** 2, dim=(0, 1))

    # Calculate RMSE per column
    columnwise_rmse = torch.sqrt(columnwise_mse)

    if scored_only:
        # The columns are:
        # 0: reactivity
        # 1: deg_Mg_pH10
        # 2: deg_pH10
        # 3: deg_Mg_50C
        # 4: deg_50C
        #
        # Scored columns are 0, 1, and 3.
        scored_indices = torch.tensor([0, 1, 3], device=inputs.device)
        columnwise_rmse = columnwise_rmse[scored_indices]

    return torch.mean(columnwise_rmse).item()
