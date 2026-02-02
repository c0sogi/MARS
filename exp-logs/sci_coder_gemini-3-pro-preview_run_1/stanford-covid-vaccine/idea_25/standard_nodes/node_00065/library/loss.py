import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Computes the Mean Squared Error (MSE) loss, masking out invalid positions.
    Used to train only on the scored positions (first 68 bases) of the RNA sequence.
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        # Use reduction='none' to get element-wise squared errors
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target, mask):
        """
        Args:
            pred (torch.Tensor): Predictions of shape (Batch, Seq_Len, Channels).
            target (torch.Tensor): Ground truth targets of shape (Batch, Seq_Len, Channels).
            mask (torch.Tensor): Boolean mask of shape (Batch, Seq_Len) where True indicates
                                 a valid position to be scored.

        Returns:
            torch.Tensor: Scalar loss value (Mean of squared errors over valid positions).
        """
        # Compute element-wise squared error: (Batch, Seq_Len, Channels)
        squared_error = self.mse(pred, target)

        # Expand mask to match the channel dimension: (Batch, Seq_Len, 1) -> (Batch, Seq_Len, Channels)
        # We cast the boolean mask to the same dtype as the predictions (float32/float16)
        mask_expanded = (
            mask.unsqueeze(-1).expand_as(squared_error).type_as(squared_error)
        )

        # Apply mask: Zero out errors for unscored positions
        masked_squared_error = squared_error * mask_expanded

        # Compute the mean over valid elements
        # Sum of errors / Number of valid elements
        # Number of valid elements = sum(mask) * num_channels
        num_valid_elements = mask_expanded.sum()

        # Avoid division by zero (though unlikely given the dataset structure)
        if num_valid_elements == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        loss = masked_squared_error.sum() / num_valid_elements

        return loss
