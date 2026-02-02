import torch
import torch.nn as nn
from library.config import Config


class MaskedHuberLoss(nn.Module):
    """
    Computes the Huber Loss (Smooth L1) with masking to ignore unscored positions.

    The loss is calculated only for positions where the mask is True.
    This handles the requirement where only the first 68 bases of the 107-length
    sequences have ground truth values.
    """

    def __init__(self, delta=1.0):
        """
        Args:
            delta (float): The threshold at which to change between delta-scaled L1 and L2 loss.
                           Default is 1.0.
        """
        super(MaskedHuberLoss, self).__init__()
        self.delta = delta
        # We use reduction='none' to obtain the element-wise loss,
        # allowing us to apply the mask manually before aggregation.
        self.loss_fn = nn.HuberLoss(reduction="none", delta=delta)

    def forward(self, preds, targets, mask):
        """
        Calculates the masked Huber loss.

        Args:
            preds (torch.Tensor): Predicted values of shape (Batch, Seq_Len, Channels).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Channels).
            mask (torch.Tensor): Boolean or binary mask of shape (Batch, Seq_Len),
                                 where 1 indicates a valid scored position.

        Returns:
            torch.Tensor: The scalar loss value averaged over valid elements.
        """
        # Ensure inputs are on the correct device
        device = preds.device

        # Compute element-wise Huber loss
        # Shape: (Batch, Seq_Len, Channels)
        raw_loss = self.loss_fn(preds, targets)

        # Expand mask to match target dimensions for broadcasting
        # Input mask: (Batch, Seq_Len)
        # Expanded mask: (Batch, Seq_Len, 1) -> Broadcasts to (Batch, Seq_Len, Channels)
        mask_expanded = mask.unsqueeze(-1).to(device).float()

        # Apply mask: Zero out loss for invalid (unscored) positions
        masked_loss = raw_loss * mask_expanded

        # Calculate the number of valid elements to normalize the loss
        # Total valid elements = (Sum of mask) * (Number of Channels)
        num_valid_elements = mask_expanded.sum() * targets.shape[-1]

        # Avoid division by zero
        if num_valid_elements == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Return mean loss over valid elements
        return masked_loss.sum() / num_valid_elements
