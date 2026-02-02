import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Computes the Mean Squared Error (MSE) between predictions and targets,
    masked to include only valid scored positions.
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, inputs, targets, mask):
        """
        Args:
            inputs (torch.Tensor): Predictions of shape (Batch, Seq_Len, Channels).
            targets (torch.Tensor): Ground truth values of shape (Batch, Seq_Len, Channels).
            mask (torch.Tensor): Boolean or float mask of shape (Batch, Seq_Len),
                                 where 1.0 indicates a valid position and 0.0 indicates
                                 an unscored/padded position.

        Returns:
            torch.Tensor: The scalar masked MSE loss.
        """
        # Calculate squared error element-wise
        # inputs, targets: (B, L, C)
        squared_error = (inputs - targets) ** 2

        # Expand mask to match channel dimension for broadcasting
        # mask: (B, L) -> (B, L, 1)
        mask_expanded = mask.unsqueeze(-1)

        # Apply mask to squared errors
        # Broadcasting (B, L, 3) * (B, L, 1) -> (B, L, 3)
        masked_squared_error = squared_error * mask_expanded

        # Calculate the number of valid elements to average over
        # We want the total count of valid entries in the (B, L, C) tensor.
        # Since the mask is 1 for valid positions, the sum of the expanded mask
        # gives the number of valid sequence positions. We multiply by the number
        # of channels to get the total number of valid scalar elements.
        num_valid_elements = mask_expanded.sum() * inputs.shape[-1]

        # Compute the mean squared error
        # Add a small epsilon to the denominator to prevent division by zero
        loss = masked_squared_error.sum() / (num_valid_elements + 1e-8)

        return loss
