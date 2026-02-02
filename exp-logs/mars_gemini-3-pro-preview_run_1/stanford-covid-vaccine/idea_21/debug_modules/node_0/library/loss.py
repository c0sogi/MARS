import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Computes Mean Squared Error (MSE) loss only on valid (masked) positions.
    Strictly applies a mask to ignore positions beyond seq_scored (index 68)
    and ensures only the scored target columns are used for gradient calculation.
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, preds, targets, mask):
        """
        Calculates the masked MSE loss.

        Args:
            preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, 3).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, 3).
            mask (torch.Tensor): Boolean mask of shape (Batch, Seq_Len), where True indicates
                                 a valid position to be scored (e.g., first 68 bases).

        Returns:
            torch.Tensor: The scalar MSE loss averaged over all valid elements.
        """
        # Ensure mask is boolean
        mask = mask.bool()

        # Expand mask to match the dimensions of predictions/targets: (B, L) -> (B, L, 3)
        # This ensures we mask all 3 target channels for invalid sequence positions.
        mask_expanded = mask.unsqueeze(-1).expand_as(preds)

        # Calculate squared differences
        diff_sq = (preds - targets) ** 2

        # Zero out the squared differences at invalid positions
        # mask_expanded.float() is 1.0 for valid, 0.0 for invalid
        masked_diff_sq = diff_sq * mask_expanded.float()

        # Compute the mean over valid elements
        # Sum of all errors divided by the number of valid elements
        sum_loss = masked_diff_sq.sum()
        count = mask_expanded.sum() + 1e-8  # Add epsilon to prevent division by zero

        return sum_loss / count


def compute_mcrmse(preds, targets, mask):
    """
    Computes Mean Columnwise Root Mean Squared Error (MCRMSE).

    This metric calculates the RMSE for each of the 3 target columns independently
    considering only the valid positions, and then averages these 3 RMSE values.
    This avoids the "Mean of Sqrts" artifact by averaging after the square root.

    Args:
        preds (torch.Tensor): Predictions of shape (Batch, Seq_Len, 3).
        targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, 3).
        mask (torch.Tensor): Boolean mask of shape (Batch, Seq_Len), where True indicates
                             a valid position.

    Returns:
        torch.Tensor: The scalar MCRMSE score.
    """
    # Ensure mask is boolean
    mask = mask.bool()

    col_rmses = []
    num_targets = preds.shape[2]  # Expected to be 3

    # Iterate over each target column (reactivity, deg_Mg_pH10, deg_Mg_50C)
    for i in range(num_targets):
        p_col = preds[:, :, i]
        t_col = targets[:, :, i]

        # Select only valid positions using the mask
        # Boolean indexing flattens the tensor to 1D, containing only valid entries
        valid_preds = p_col[mask]
        valid_targets = t_col[mask]

        # Compute MSE for this specific column
        mse = torch.mean((valid_preds - valid_targets) ** 2)

        # Compute RMSE for this specific column
        rmse = torch.sqrt(mse)
        col_rmses.append(rmse)

    # Average the RMSEs across the 3 columns
    return torch.mean(torch.stack(col_rmses))
