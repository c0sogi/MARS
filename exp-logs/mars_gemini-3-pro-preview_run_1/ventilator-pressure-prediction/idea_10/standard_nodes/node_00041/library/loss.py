import torch
import torch.nn as nn
from library.config import Config


class MaskedL1Loss(nn.Module):
    """
    Computes the Mean Absolute Error (L1 Loss) strictly for the inspiratory phase.
    The inspiratory phase is defined where the control input u_out is 0.
    """

    def __init__(self):
        super(MaskedL1Loss, self).__init__()
        self.l1 = nn.L1Loss(reduction="none")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, u_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Model predictions. Shape (Batch, Seq_Len) or (Batch, Seq_Len, 1).
            target: Ground truth pressure. Shape matches pred.
            u_out: Control input indicating phase (0=Inspiratory, 1=Expiratory).
                   Shape matches pred or is broadcastable.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure shapes align
        if target.shape != pred.shape:
            target = target.view_as(pred)
        if u_out.shape != pred.shape:
            u_out = u_out.view_as(pred)

        # Create mask: 1 for inspiratory (u_out == 0), 0 for expiratory
        mask = 1.0 - u_out

        # Compute element-wise L1 loss
        loss_elements = self.l1(pred, target)

        # Apply mask
        masked_loss = loss_elements * mask

        # Compute mean over valid (inspiratory) elements
        # Add a small epsilon to denominator to prevent division by zero in edge cases
        sum_mask = torch.sum(mask)
        if sum_mask < 1e-8:
            # If no inspiratory phase exists in batch (unlikely), return 0 loss with grad
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        loss = torch.sum(masked_loss) / sum_mask
        return loss


class CompositeLoss(nn.Module):
    """
    Implements the Deep Supervision loss.
    Combines the loss from the final output layer and the auxiliary intermediate layer.

    Formula: Loss = L_mask(final) + weight * L_mask(aux)
    """

    def __init__(self):
        super(CompositeLoss, self).__init__()
        self.masked_l1 = MaskedL1Loss()
        self.aux_weight = Config.AUX_WEIGHT

    def forward(
        self, outputs, target: torch.Tensor, u_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            outputs: Tuple (final_pred, aux_pred) from the model, or single tensor.
            target: Ground truth pressure.
            u_out: Control input for masking.

        Returns:
            torch.Tensor: Weighted composite loss.
        """
        # Check if outputs is a tuple (Deep Supervision enabled)
        if isinstance(outputs, (tuple, list)):
            final_pred, aux_pred = outputs

            # Calculate loss for final head
            loss_final = self.masked_l1(final_pred, target, u_out)

            # Calculate loss for auxiliary head
            loss_aux = self.masked_l1(aux_pred, target, u_out)

            # Weighted sum
            return loss_final + (self.aux_weight * loss_aux)
        else:
            # Fallback for inference or simple validation where only final output is returned
            return self.masked_l1(outputs, target, u_out)
