import torch
import torch.nn as nn


class MaskedL1Loss(nn.Module):
    """
    Custom loss function for the Ventilator Pressure Prediction task.

    Implements a Masked Mean Absolute Error (L1) loss that:
    1. Only calculates error during the inspiratory phase (u_out == 0).
    2. Combines the loss from the final prediction head with the auxiliary
       prediction head (Deep Supervision) using a weighted sum.

    Formula: Loss = MAE(final) + aux_weight * MAE(aux)
    """

    def __init__(self, aux_weight: float = 0.3):
        """
        Args:
            aux_weight (float): Weight applied to the auxiliary head's loss.
        """
        super().__init__()
        self.aux_weight = aux_weight
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, final_pred, aux_pred, target, u_out):
        """
        Calculates the weighted masked L1 loss.

        Args:
            final_pred (torch.Tensor): Final model predictions of shape (Batch, Seq, 1).
            aux_pred (torch.Tensor or None): Auxiliary model predictions of shape (Batch, Seq, 1).
                                             Can be None during inference/validation.
            target (torch.Tensor): Ground truth pressure of shape (Batch, Seq) or (Batch, Seq, 1).
            u_out (torch.Tensor): Control input 'u_out' of shape (Batch, Seq) or (Batch, Seq, 1).
                                  Values are 0 (inspiration) or 1 (expiration).

        Returns:
            torch.Tensor: The scalar total loss.
        """
        # 1. Align shapes: Ensure target and u_out are (Batch, Seq, 1)
        if target.dim() == 2:
            target = target.unsqueeze(-1)

        if u_out.dim() == 2:
            u_out = u_out.unsqueeze(-1)

        # 2. Create Mask
        # We want to score ONLY the inspiratory phase where u_out == 0.
        # Mask = 1 when u_out is 0, Mask = 0 when u_out is 1.
        mask = 1.0 - u_out

        # Calculate normalization factor (number of valid time steps)
        # Add epsilon to prevent division by zero in the unlikely case of a full-expiratory batch
        mask_sum = mask.sum() + 1e-8

        # 3. Calculate Final Head Loss
        loss_final_elementwise = self.l1(final_pred, target)
        loss_final_masked = loss_final_elementwise * mask
        loss_final = loss_final_masked.sum() / mask_sum

        # 4. Calculate Auxiliary Head Loss (if available)
        loss_aux = torch.tensor(0.0, device=final_pred.device, dtype=final_pred.dtype)
        if aux_pred is not None:
            loss_aux_elementwise = self.l1(aux_pred, target)
            loss_aux_masked = loss_aux_elementwise * mask
            loss_aux = loss_aux_masked.sum() / mask_sum

        # 5. Combine Losses
        total_loss = loss_final + (self.aux_weight * loss_aux)

        return total_loss
