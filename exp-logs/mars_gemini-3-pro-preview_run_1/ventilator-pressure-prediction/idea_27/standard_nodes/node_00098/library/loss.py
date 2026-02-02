import torch
import torch.nn as nn
from library.config import Config


class MaskedAuxL1Loss(nn.Module):
    """
    Computes the weighted Masked L1 Loss for the Wide-Projected Deeply-Supervised Network.

    Logic:
    1. Calculates L1 loss (Mean Absolute Error).
    2. Applies a mask to ignore the expiratory phase (where u_out == 1).
    3. Computes loss for both the final head and the auxiliary head.
    4. Returns the weighted sum: L_total = L_final + aux_weight * L_aux.
    """

    def __init__(self):
        super(MaskedAuxL1Loss, self).__init__()
        self.aux_weight = Config.aux_weight

    def forward(self, preds, target, u_out):
        """
        Args:
            preds (tuple): (final_pred, aux_pred) from the model.
                           Shapes are expected to be (Batch, Seq, 1) or (Batch, Seq).
            target (Tensor): Ground truth pressure. Shape (Batch, Seq).
            u_out (Tensor): Expiratory control input (1=expiratory, 0=inspiratory).
                            Shape (Batch, Seq).

        Returns:
            Tensor: Scalar loss value.
        """
        final_pred, aux_pred = preds

        # Ensure predictions match target shape (Batch, Seq)
        if final_pred.dim() == 3:
            final_pred = final_pred.squeeze(-1)
        if aux_pred.dim() == 3:
            aux_pred = aux_pred.squeeze(-1)

        # Calculate individual losses
        loss_final = self._masked_mae(final_pred, target, u_out)
        loss_aux = self._masked_mae(aux_pred, target, u_out)

        # Combine losses
        total_loss = loss_final + self.aux_weight * loss_aux

        return total_loss

    def _masked_mae(self, pred, target, u_out):
        """
        Internal helper to calculate MAE for the inspiratory phase only.
        """
        # u_out is 1 for expiratory, 0 for inspiratory.
        # We want to score inspiratory, so mask weight = 1 - u_out.
        mask = 1 - u_out

        # Calculate absolute error
        error = torch.abs(pred - target)

        # Apply mask
        masked_error = error * mask

        # Normalize by the number of valid (inspiratory) time steps
        sum_mask = mask.sum()

        # Handle edge case where batch has no inspiratory phase (unlikely but safe)
        if sum_mask < 1e-6:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return masked_error.sum() / sum_mask
