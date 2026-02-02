import torch
import torch.nn as nn
from library.config import Config


class MaskedL1Loss(nn.Module):
    """
    Implements the Weighted Masked L1 Loss.
    Calculates Mean Absolute Error (MAE) strictly during the inspiratory phase (u_out == 0).
    Combines the final head loss with the auxiliary head loss using a defined weight.
    """

    def __init__(self):
        super().__init__()
        self.aux_weight = Config.LOSS_AUX_WEIGHT
        self.loss_fn = nn.L1Loss(reduction="mean")

    def _compute_phase_loss(self, pred, target, mask):
        """
        Computes L1 loss for a single prediction tensor, masked by the inspiratory phase.

        Args:
            pred (torch.Tensor): Prediction tensor of shape (B, S, 1).
            target (torch.Tensor): Target tensor of shape (B, S, 1).
            mask (torch.Tensor): Boolean mask tensor of shape (B, S, 1), True for inspiratory phase.

        Returns:
            torch.Tensor: Scalar L1 loss.
        """
        # Select only the elements corresponding to the inspiratory phase
        pred_insp = torch.masked_select(pred, mask)
        target_insp = torch.masked_select(target, mask)

        # Compute MAE on the valid elements
        # If mask is empty (unlikely in this dataset), this returns NaN, which is handled by gradients usually
        if pred_insp.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return self.loss_fn(pred_insp, target_insp)

    def forward(self, preds, target, u_out):
        """
        Args:
            preds (tuple): Tuple containing (final_pred, aux_pred).
                           Each is a tensor of shape (Batch, Seq, 1).
            target (torch.Tensor): Ground truth pressure of shape (Batch, Seq).
            u_out (torch.Tensor): Control input u_out of shape (Batch, Seq).

        Returns:
            torch.Tensor: The calculated weighted loss.
        """
        final_pred, aux_pred = preds

        # Ensure target and u_out have the channel dimension to match predictions (B, S, 1)
        if target.dim() == 2:
            target = target.unsqueeze(-1)
        if u_out.dim() == 2:
            u_out = u_out.unsqueeze(-1)

        # Create boolean mask for inspiratory phase (u_out == 0)
        # Using strict equality as u_out is binary (0 or 1)
        mask = u_out == 0

        # 1. Calculate Loss for Final Head
        loss_final = self._compute_phase_loss(final_pred, target, mask)

        # 2. Calculate Loss for Auxiliary Head (if it exists)
        loss_aux = torch.tensor(0.0, device=final_pred.device)
        if aux_pred is not None:
            loss_aux = self._compute_phase_loss(aux_pred, target, mask)

        # 3. Combine Losses
        total_loss = loss_final + (self.aux_weight * loss_aux)

        return total_loss
