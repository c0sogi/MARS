import torch
import torch.nn as nn
from library.config import Config


class WeightedL1Loss(nn.Module):
    """
    Custom Loss function for Ventilator Pressure Prediction.

    Implements a weighted L1 loss (Mean Absolute Error) where the inspiratory phase
    (u_out=0) is weighted higher than the expiratory phase (u_out=1).

    Strategy: 'Stretched-Horizon Convergence Protocol'
    - Inspiratory Weight: 1.0 (Focus on competition metric)
    - Expiratory Weight: 0.1 (Maintain temporal context without overfitting irrelevant phase)
    """

    def __init__(self):
        super(WeightedL1Loss, self).__init__()
        self.w_insp = Config.INSPIRATORY_WEIGHT
        self.w_exp = Config.EXPIRATORY_WEIGHT

    def forward(self, preds, targets, u_out):
        """
        Calculates the weighted mean absolute error.

        Args:
            preds (torch.Tensor): Model predictions. Shape: (Batch, Seq) or (Batch, Seq, 1)
            targets (torch.Tensor): Ground truth pressure values. Shape matches preds.
            u_out (torch.Tensor): Binary control input indicating expiratory phase (1) or inspiratory (0).

        Returns:
            torch.Tensor: Scalar loss value (mean of weighted errors).
        """
        # Ensure u_out is the same shape as preds for broadcasting
        if u_out.shape != preds.shape:
            u_out = u_out.view_as(preds)

        # Ensure u_out is float for calculation
        u_out = u_out.float()

        # Calculate element-wise absolute error (L1)
        abs_error = torch.abs(preds - targets)

        # Generate weights based on u_out status
        # If u_out == 0 (Inspiratory) -> weight = w_insp
        # If u_out == 1 (Expiratory)  -> weight = w_exp
        weights = (1.0 - u_out) * self.w_insp + u_out * self.w_exp

        # Apply weights to the error
        weighted_error = abs_error * weights

        # Return the mean loss
        return weighted_error.mean()


def competition_metric(preds, targets, u_out):
    """
    Calculates the official competition metric: Mean Absolute Error (MAE)
    calculated ONLY during the inspiratory phase (u_out == 0).

    Args:
        preds (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth pressure values.
        u_out (torch.Tensor): Binary control input.

    Returns:
        float: The MAE score for the inspiratory phase.
    """
    # Ensure u_out is aligned
    if u_out.shape != preds.shape:
        u_out = u_out.view_as(preds)

    # Create boolean mask for inspiratory phase (u_out == 0)
    # Using < 0.5 to handle potential float representations of binary flags safely
    mask = u_out < 0.5

    # Select only inspiratory phase elements
    insp_preds = torch.masked_select(preds, mask)
    insp_targets = torch.masked_select(targets, mask)

    # Calculate MAE
    if insp_preds.numel() == 0:
        return 0.0

    mae = torch.abs(insp_preds - insp_targets).mean()

    return mae.item()
