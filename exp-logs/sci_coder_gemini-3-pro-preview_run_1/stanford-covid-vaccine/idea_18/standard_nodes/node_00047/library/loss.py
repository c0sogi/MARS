import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Computes the Mean Squared Error (MSE) loss, masking out invalid positions.
    Used for training the RNA degradation model on the 68 scored positions.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, preds, targets, mask):
        """
        Args:
            preds (torch.Tensor): Predictions of shape (Batch, SeqLen, Channels).
            targets (torch.Tensor): Ground truth of shape (Batch, SeqLen, Channels).
            mask (torch.Tensor): Boolean mask of shape (Batch, SeqLen), True for valid positions.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Compute element-wise squared error: (B, L, C)
        loss = self.mse(preds, targets)

        # Expand mask to match channel dimension: (B, L) -> (B, L, 1)
        mask_expanded = mask.unsqueeze(-1)

        # Apply mask to zero out loss at invalid positions
        masked_loss = loss * mask_expanded.float()

        # Calculate the number of valid elements (N_valid_positions * N_channels)
        num_valid = mask_expanded.sum() * preds.shape[-1]

        # Avoid division by zero
        if num_valid == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        # Return mean loss over valid elements
        return masked_loss.sum() / num_valid


def mcrmse(preds, targets, mask):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE).

    Metric Definition:
    1. Filter predictions and targets to valid positions only.
    2. Calculate RMSE separately for each target column.
    3. Take the average of these RMSE values.

    Args:
        preds (torch.Tensor): Predictions of shape (Batch, SeqLen, Channels).
        targets (torch.Tensor): Ground truth of shape (Batch, SeqLen, Channels).
        mask (torch.Tensor): Boolean mask of shape (Batch, SeqLen).

    Returns:
        torch.Tensor: Scalar MCRMSE value.
    """
    # Select valid predictions and targets using the mask.
    # This flattens the batch and sequence dimensions, resulting in (N_total_valid, Channels)
    valid_preds = preds[mask]
    valid_targets = targets[mask]

    # Calculate MSE for each column independently (dim=0 is the flattened sample dim)
    mse_per_col = torch.mean((valid_preds - valid_targets) ** 2, dim=0)

    # Calculate RMSE for each column
    rmse_per_col = torch.sqrt(mse_per_col)

    # Calculate the mean of the column-wise RMSEs
    mcrmse_val = torch.mean(rmse_per_col)

    return mcrmse_val
