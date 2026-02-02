import torch
import torch.nn as nn


class MaskedAuxiliaryLoss(nn.Module):
    """
    Computes the weighted Masked L1 Loss for the Ventilator Pressure Prediction task.

    The metric is Mean Absolute Error (MAE) calculated only during the inspiratory phase
    of the breath. The inspiratory phase is defined where the exploratory valve is closed
    (u_out == 0).

    The total loss is a weighted sum of the loss from the final output head and the
    auxiliary output head (if present).

    Formula: Total Loss = Loss_final + (aux_weight * Loss_aux)
    """

    def __init__(self, aux_weight: float = 0.3):
        """
        Initialize the loss function.

        Args:
            aux_weight (float): The weight coefficient for the auxiliary loss term.
                                Defaults to 0.3 as per the strategy.
        """
        super().__init__()
        self.aux_weight = aux_weight

    def forward(self, preds, target, u_out):
        """
        Compute the masked loss.

        Args:
            preds (tuple): A tuple containing (final_pred, aux_pred).
                - final_pred (torch.Tensor): Predictions from the main head. Shape (Batch, Length).
                - aux_pred (torch.Tensor or None): Predictions from the auxiliary head. Shape (Batch, Length).
            target (torch.Tensor): Ground truth pressure values. Shape (Batch, Length).
            u_out (torch.Tensor): Control input indicating the valve state (0 or 1). Shape (Batch, Length).

        Returns:
            torch.Tensor: The computed scalar total loss.
        """
        final_pred, aux_pred = preds

        # Generate the mask for the inspiratory phase.
        # We want to score when u_out == 0, so mask = 1 - u_out.
        mask = 1.0 - u_out

        # Calculate the number of valid steps for normalization.
        # Adding a small epsilon to prevent division by zero in edge cases.
        mask_sum = mask.sum() + 1e-8

        # --- Calculate Final Head Loss ---
        # Compute element-wise L1 error
        mae_final = torch.abs(final_pred - target)
        # Apply mask and normalize
        loss_final = torch.sum(mae_final * mask) / mask_sum

        # --- Calculate Auxiliary Head Loss ---
        loss_aux = 0.0
        if aux_pred is not None:
            mae_aux = torch.abs(aux_pred - target)
            loss_aux = torch.sum(mae_aux * mask) / mask_sum

        # --- Combine Losses ---
        total_loss = loss_final + (self.aux_weight * loss_aux)

        return total_loss
