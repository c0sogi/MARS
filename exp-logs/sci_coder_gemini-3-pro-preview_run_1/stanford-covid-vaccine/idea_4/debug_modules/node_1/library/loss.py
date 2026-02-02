import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Custom loss function for RNA degradation prediction.
    Calculates Mean Squared Error (MSE) only on valid (scored) positions defined by a binary mask.
    This ensures the model is optimized based on the 68 scored positions while processing
    the full 107-length sequence context.
    """

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, inputs, targets, mask):
        """
        Forward pass for the Masked MSE Loss.

        Args:
            inputs (torch.Tensor): Predicted values from the model.
                                   Shape: (Batch, Seq_Len, Num_Targets)
            targets (torch.Tensor): Ground truth target values.
                                    Shape: (Batch, Seq_Len, Num_Targets)
            mask (torch.Tensor): Binary mask indicating valid positions.
                                 Shape: (Batch, Seq_Len)
                                 Values should be 1.0 for scored positions and 0.0 for unscored/padded positions.

        Returns:
            torch.Tensor: The calculated scalar loss (Mean Squared Error over valid elements).
        """
        # 1. Calculate element-wise squared error
        # Shape: (Batch, Seq_Len, Num_Targets)
        squared_diff = (inputs - targets) ** 2

        # 2. Expand mask to broadcast over the target dimension
        # Current mask shape: (Batch, Seq_Len)
        # Target shape: (Batch, Seq_Len, 1) to match (Batch, Seq_Len, Num_Targets)
        mask_expanded = mask.unsqueeze(-1)

        # 3. Apply the mask to the squared errors
        # Positions with mask=0 will become 0 and not contribute to the loss sum
        masked_loss = squared_diff * mask_expanded

        # 4. Calculate the normalization factor (total number of valid scalar predictions)
        # mask.sum() gives the total number of valid sequence positions in the batch
        # We multiply by inputs.shape[-1] (number of target columns) to get total valid elements
        num_valid_elements = mask.sum() * inputs.shape[-1]

        # 5. Compute the mean loss
        # We sum all masked errors and divide by the number of valid elements.
        # A small epsilon (1e-8) is added to the denominator to prevent division by zero.
        loss = masked_loss.sum() / (num_valid_elements + 1e-8)

        return loss
