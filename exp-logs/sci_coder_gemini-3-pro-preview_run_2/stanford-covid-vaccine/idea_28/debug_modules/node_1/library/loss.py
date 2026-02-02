import torch
import torch.nn as nn
from library.config import Config, mcrmse_loss


class MCRMSELoss(nn.Module):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE)
    specifically for the three scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
    while masking out padding and unscored positions.
    """

    def __init__(self):
        super().__init__()
        self.scored_length = Config.SCORED_LENGTH

    def forward(self, preds, targets, mask=None):
        """
        Computes the MCRMSE loss.

        Args:
            preds (torch.Tensor): Predicted values of shape (B, L, 5).
            targets (torch.Tensor): Ground truth values of shape (B, L, 5).
            mask (torch.Tensor, optional): Mask of shape (B, L) indicating valid positions.
                                           If None, a mask is automatically generated based on
                                           Config.SCORED_LENGTH.

        Returns:
            torch.Tensor: The scalar MCRMSE loss.
        """
        # If no mask is provided, create one that selects the first SCORED_LENGTH positions
        if mask is None:
            B, L, _ = preds.shape
            mask = torch.zeros((B, L), device=preds.device, dtype=torch.float32)
            mask[:, : self.scored_length] = 1.0

        # Delegate calculation to the provided library function
        return mcrmse_loss(preds, targets, mask)
