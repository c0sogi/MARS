import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """
    Implements Masked Mean Squared Error Loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred (torch.Tensor): Predicted values. Shape [Batch, SeqLen, 3].
            target (torch.Tensor): Ground truth values. Shape [Batch, SeqLen, 3].
            mask (torch.Tensor): Binary mask. Shape [Batch, SeqLen].
        """
        # Expand mask to match channel dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(pred)

        # Calculate number of valid elements
        count = mask_expanded.sum() + 1e-8

        # Calculate squared difference
        diff = pred - target

        # Apply mask
        masked_diff = diff * mask_expanded

        # Sum of squares divided by count = MSE
        loss = torch.sum(masked_diff**2) / count

        return loss
