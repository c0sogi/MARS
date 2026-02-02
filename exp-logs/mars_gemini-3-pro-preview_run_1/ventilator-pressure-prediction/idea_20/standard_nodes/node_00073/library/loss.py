import torch
import torch.nn as nn


class MaskedL1Loss(nn.Module):
    """
    Custom L1 Loss that ignores the expiratory phase of the breath.
    The expiratory phase is indicated by u_out=1.
    The inspiratory phase (to be scored) is indicated by u_out=0.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, u_out):
        """
        Computes the masked Mean Absolute Error.

        Args:
            pred (torch.Tensor): Predicted pressure values.
                                 Shape can be (Batch, Seq_Len) or (Batch, Seq_Len, 1).
            target (torch.Tensor): Actual pressure values.
                                   Shape (Batch, Seq_Len).
            u_out (torch.Tensor): Control input indicating phase.
                                  Shape (Batch, Seq_Len).
                                  0.0 = Inspiratory (Include), 1.0 = Expiratory (Exclude).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Flatten all tensors to ensure shape alignment (N*L,)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        u_out_flat = u_out.view(-1)

        # Create the mask: we want to keep u_out == 0
        # Since u_out is binary (0 or 1), 1 - u_out gives 1 for inspiratory, 0 for expiratory
        mask = 1.0 - u_out_flat

        # Calculate absolute error
        abs_error = torch.abs(pred_flat - target_flat)

        # Apply the mask to the errors
        masked_error = abs_error * mask

        # Calculate the mean error only over the valid (masked) elements
        # We divide by the sum of the mask to get the mean over inspiratory steps
        # Add epsilon to avoid division by zero
        loss = masked_error.sum() / (mask.sum() + 1e-8)

        return loss
