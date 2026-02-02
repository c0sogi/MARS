import torch
import torch.nn as nn
from library.config import Config


class WeightedL1Loss(nn.Module):
    """
    Custom Weighted L1 Loss function for the Ventilator Pressure Prediction task.

    This loss function applies different weights to the inspiratory and expiratory phases
    of the breath. The competition metric is calculated only on the inspiratory phase,
    so a higher weight is assigned to it (default 1.0). A non-zero weight (default 0.1)
    is assigned to the expiratory phase to ensure the Recurrent Neural Network maintains
    valid hidden states and temporal context throughout the entire sequence, preventing
    instability during the transition between breaths or phases.
    """

    def __init__(
        self,
        inspiratory_weight=Config.LOSS_INSPIRATORY_WEIGHT,
        expiratory_weight=Config.LOSS_EXPIRATORY_WEIGHT,
    ):
        """
        Initialize the WeightedL1Loss.

        Args:
            inspiratory_weight (float): Weight applied to the inspiratory phase (u_out=0).
                                        Defaults to Config.LOSS_INSPIRATORY_WEIGHT.
            expiratory_weight (float): Weight applied to the expiratory phase (u_out=1).
                                       Defaults to Config.LOSS_EXPIRATORY_WEIGHT.
        """
        super(WeightedL1Loss, self).__init__()
        self.inspiratory_weight = inspiratory_weight
        self.expiratory_weight = expiratory_weight

    def forward(self, preds, targets, u_out):
        """
        Calculate the weighted L1 loss.

        Args:
            preds (torch.Tensor): Predicted pressure values. Shape: (Batch, Seq_Len) or (N,).
            targets (torch.Tensor): Actual pressure values. Shape must match preds.
            u_out (torch.Tensor): Control input indicating the phase of the breath.
                                  0 for inspiratory, 1 for expiratory.
                                  Shape must match preds.

        Returns:
            torch.Tensor: The scalar weighted mean absolute error.
        """
        # Calculate element-wise absolute error
        abs_error = torch.abs(preds - targets)

        # Determine weights based on u_out
        # u_out is 0 for inspiratory, 1 for expiratory.
        # We use a threshold of 0.5 to handle potential float representations of the binary flag.
        weights = torch.where(
            u_out > 0.5,
            torch.tensor(
                self.expiratory_weight, device=u_out.device, dtype=preds.dtype
            ),
            torch.tensor(
                self.inspiratory_weight, device=u_out.device, dtype=preds.dtype
            ),
        )

        # Apply weights
        weighted_error = abs_error * weights

        # Return the mean loss
        return weighted_error.mean()
