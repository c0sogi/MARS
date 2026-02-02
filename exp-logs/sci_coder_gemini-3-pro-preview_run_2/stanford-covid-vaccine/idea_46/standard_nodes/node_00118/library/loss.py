import torch
import torch.nn as nn
from library.config import Config


def mcrmse_loss(pred, target, mask=None):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    This function computes the RMSE for specific columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
    and specific sequence positions (0 to 67), ignoring the rest.

    Args:
        pred (torch.Tensor): Predictions of shape (B, L, 5).
        target (torch.Tensor): Ground truth targets of shape (B, L, 5).
        mask (torch.Tensor, optional): Mask tensor. Not strictly used here as we
                                       slice by fixed Config.PRED_LEN, but kept for API consistency.

    Returns:
        torch.Tensor: The scalar MCRMSE loss.
    """
    # Indices of the columns to be scored: [0, 1, 3] corresponding to reactivity, deg_Mg_pH10, deg_Mg_50C
    scored_cols = Config.TARGET_INDICES

    # The length of the sequence to score (first 68 bases)
    pred_len = Config.PRED_LEN

    loss = 0.0
    count = 0

    for col_idx in scored_cols:
        # Slice the predictions and targets to the scored length and specific column
        # Shape becomes (B, pred_len)
        p = pred[:, :pred_len, col_idx]
        t = target[:, :pred_len, col_idx]

        # Calculate Mean Squared Error for this column
        mse = (p - t) ** 2

        # Calculate Root Mean Squared Error
        # Note: mean() averages over the batch and the sequence length
        root_mean_mse = torch.sqrt(mse.mean())

        loss += root_mean_mse
        count += 1

    # Return the mean of the RMSEs across the scored columns
    return loss / count


class MCRMSELoss(nn.Module):
    """
    PyTorch Module wrapper for the MCRMSE loss function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask=None):
        return mcrmse_loss(pred, target, mask)
