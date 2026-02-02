import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Custom loss function that calculates Mean Squared Error (MSE) only for
    specific positions defined by a binary mask.

    This ensures gradients are calculated only for the first 68 positions
    (seq_scored) as specified in the task description, ignoring the unscored
    tail of the sequence.
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        # We use reduction='none' to calculate (y_pred - y_true)^2 for every element
        # so we can apply the mask before averaging.
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, inputs, targets, mask):
        """
        Args:
            inputs (torch.Tensor): Predictions from the model.
                                   Shape: (Batch, Seq_Len, Channels)
            targets (torch.Tensor): Ground truth values.
                                    Shape: (Batch, Seq_Len, Channels)
            mask (torch.Tensor): Binary mask indicating valid positions.
                                 Shape: (Batch, Seq_Len)
                                 1.0 indicates a scored position, 0.0 otherwise.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # 1. Calculate element-wise squared errors
        # Shape: (Batch, Seq_Len, Channels)
        raw_loss = self.mse(inputs, targets)

        # 2. Expand mask to match dimensions
        # The mask is (Batch, Seq_Len), but targets are (Batch, Seq_Len, Channels).
        # We unsqueeze to get (Batch, Seq_Len, 1), allowing broadcasting over channels.
        mask_expanded = mask.unsqueeze(-1)

        # 3. Apply mask
        # Zero out the loss for positions that are not scored.
        masked_loss = raw_loss * mask_expanded

        # 4. Compute the mean over valid elements
        # Count total number of valid entries (positions * channels)
        num_valid_elements = mask_expanded.sum()

        # Avoid division by zero (though seq_scored is guaranteed > 0 in this dataset)
        if num_valid_elements > 0:
            loss = masked_loss.sum() / num_valid_elements
        else:
            loss = masked_loss.sum() * 0.0

        return loss
