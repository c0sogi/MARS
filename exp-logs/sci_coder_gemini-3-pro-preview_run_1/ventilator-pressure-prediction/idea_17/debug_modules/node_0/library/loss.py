import torch
import torch.nn as nn
from library.config import Config


class CompositeMaskedL1Loss(nn.Module):
    """
    Custom loss function for the Wide-State Composite Network.
    Computes the Mean Absolute Error (MAE) masked by the inspiratory phase (u_out == 0).
    Combines the loss from the final prediction head and the auxiliary deep supervision head.
    """

    def __init__(self):
        """
        Initialize the loss function.
        Retrieves the auxiliary loss weight from the global configuration.
        """
        super().__init__()
        self.aux_weight = Config.AUXILIARY_WEIGHT

    def masked_mae(self, pred, target, u_out):
        """
        Calculates Mean Absolute Error only for time steps where u_out == 0.

        Args:
            pred (torch.Tensor): Predicted values of shape (B, L).
            target (torch.Tensor): Ground truth values of shape (B, L).
            u_out (torch.Tensor): Control flag of shape (B, L), where 1 indicates expiratory phase.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Create mask: 1 where u_out is 0 (inspiratory), 0 otherwise
        mask = 1 - u_out

        # Calculate absolute error
        error = torch.abs(pred - target)

        # Apply mask to error
        masked_error = error * mask

        # Calculate mean over the valid elements
        # Add epsilon to denominator to prevent division by zero
        loss = masked_error.sum() / (mask.sum() + 1e-8)

        return loss

    def forward(self, preds, target, u_out):
        """
        Computes the composite loss.

        Args:
            preds (tuple): Tuple containing (final_pred, aux_pred).
                           final_pred: Tensor of shape (B, L).
                           aux_pred: Tensor of shape (B, L) or None.
            target (torch.Tensor): Ground truth pressure of shape (B, L).
            u_out (torch.Tensor): Control flag of shape (B, L).

        Returns:
            torch.Tensor: The weighted sum of the final and auxiliary losses.
        """
        final_pred, aux_pred = preds

        # Calculate loss for the final head
        loss_final = self.masked_mae(final_pred, target, u_out)

        # Calculate loss for the auxiliary head if it exists and weight is positive
        if aux_pred is not None and self.aux_weight > 0:
            loss_aux = self.masked_mae(aux_pred, target, u_out)
            total_loss = loss_final + self.aux_weight * loss_aux
        else:
            total_loss = loss_final

        return total_loss
