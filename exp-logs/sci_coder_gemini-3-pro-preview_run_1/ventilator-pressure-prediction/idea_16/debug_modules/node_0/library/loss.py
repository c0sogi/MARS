import torch
import torch.nn as nn
from library.config import AUX_WEIGHT


class MaskedL1Loss(nn.Module):
    """
    Custom objective function for Ventilator Pressure Prediction.
    Calculates Mean Absolute Error (MAE) strictly for the inspiratory phase (u_out == 0).
    Computes a composite loss including an auxiliary head prediction.
    """

    def __init__(self, aux_weight: float = AUX_WEIGHT):
        """
        Args:
            aux_weight (float): Weight applied to the auxiliary loss term.
                                Defaults to configuration value.
        """
        super(MaskedL1Loss, self).__init__()
        self.aux_weight = aux_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        u_out: torch.Tensor,
        aux_pred: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculates the weighted composite masked L1 loss.

        Args:
            pred (torch.Tensor): Main predictions from the model.
                                 Shape: (Batch, Seq_Len, 1) or (Batch, Seq_Len).
            target (torch.Tensor): Ground truth pressure values.
                                   Shape: (Batch, Seq_Len).
            u_out (torch.Tensor): Control input indicating expiratory phase (1) or inspiratory (0).
                                  Shape: (Batch, Seq_Len).
            aux_pred (torch.Tensor, optional): Auxiliary predictions from the model (Deep Supervision).
                                               Shape: (Batch, Seq_Len, 1) or (Batch, Seq_Len).

        Returns:
            torch.Tensor: The scalar composite loss value.
        """
        # Create mask: 1 where u_out == 0 (inspiration), 0 otherwise
        mask = 1 - u_out

        # Calculate normalization factor (number of valid inspiratory time steps)
        # Add epsilon to prevent division by zero in case of empty mask (unlikely but safe)
        normalization = mask.sum() + 1e-8

        # Helper function to compute masked MAE
        def compute_masked_mae(p, t, m, norm):
            # Squeeze last dimension if prediction is (Batch, Seq, 1)
            if p.dim() > t.dim():
                p = p.squeeze(-1)

            # Calculate absolute error, apply mask, sum and normalize
            return (torch.abs(p - t) * m).sum() / norm

        # Calculate Main Loss
        loss_main = compute_masked_mae(pred, target, mask, normalization)

        # Calculate Total Loss
        total_loss = loss_main

        # Add Auxiliary Loss if provided
        if aux_pred is not None:
            loss_aux = compute_masked_mae(aux_pred, target, mask, normalization)
            total_loss += self.aux_weight * loss_aux

        return total_loss
