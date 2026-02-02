import torch
import torch.nn as nn
import numpy as np
from library.config import Config


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) loss.
    Used as the training objective.

    Equation:
    MCRMSE = (1/Nt) * sum_j( sqrt( (1/n) * sum_i( (y_ij - y_hat_ij)^2 ) ) )

    Where:
        Nt = Number of target columns (5)
        n = Total number of scored positions (Batch Size * Scored Sequence Length)
    """

    def __init__(self):
        super(MCRMSELoss, self).__init__()
        self.scored_len = Config.PRED_LEN

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Model predictions of shape (N, 107, 5).
            targets (Tensor): Ground truth values of shape (N, 68, 5).

        Returns:
            Tensor: Scalar loss value.
        """
        # Slice inputs to match the scored length of targets (first 68 positions)
        # inputs: (Batch, Seq_Len_Full, Channels) -> (Batch, Seq_Len_Scored, Channels)
        inputs_sliced = inputs[:, : self.scored_len, :]

        # Calculate Squared Error: (y - y_hat)^2
        squared_diff = (inputs_sliced - targets) ** 2

        # Calculate Mean Squared Error per column (averaging over Batch and Sequence dimensions)
        # dim 0 is Batch, dim 1 is Sequence. Result shape: (5,)
        mse_per_column = torch.mean(squared_diff, dim=(0, 1))

        # Calculate RMSE per column
        rmse_per_column = torch.sqrt(mse_per_column)

        # Calculate Mean of RMSEs across all columns
        loss = torch.mean(rmse_per_column)

        return loss


def scored_mcrmse(preds, truths):
    """
    Calculates the MCRMSE specifically for the competition metric.

    Logic:
    1. Slices predictions to the first 68 positions.
    2. Filters for the 3 scored columns: reactivity, deg_Mg_pH10, deg_Mg_50C.
    3. Computes MCRMSE.

    Args:
        preds (Tensor or np.ndarray): Predictions of shape (N, 107, 5).
        truths (Tensor or np.ndarray): Ground truth of shape (N, 68, 5).

    Returns:
        float: The calculated MCRMSE score.
    """
    # Convert to PyTorch tensors if input is numpy
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(truths, np.ndarray):
        truths = torch.from_numpy(truths)

    # Ensure inputs are float
    preds = preds.float()
    truths = truths.float()

    # 1. Slice predictions to scored length (68)
    preds_sliced = preds[:, : Config.PRED_LEN, :]

    # Ensure truths are also sliced/correct length (though they should be 68 already)
    truths_sliced = truths[:, : Config.PRED_LEN, :]

    # 2. Identify indices of the scored columns
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Config.SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]

    all_cols = Config.TARGET_COLS
    scored_cols = set(Config.SCORED_COLS)

    scored_indices = [i for i, col in enumerate(all_cols) if col in scored_cols]

    # Select only the scored columns
    # Shape becomes (N, 68, 3)
    preds_filtered = preds_sliced[:, :, scored_indices]
    truths_filtered = truths_sliced[:, :, scored_indices]

    # 3. Compute MCRMSE
    squared_diff = (preds_filtered - truths_filtered) ** 2
    mse_per_column = torch.mean(squared_diff, dim=(0, 1))
    rmse_per_column = torch.sqrt(mse_per_column)
    score = torch.mean(rmse_per_column)

    return score.item()
