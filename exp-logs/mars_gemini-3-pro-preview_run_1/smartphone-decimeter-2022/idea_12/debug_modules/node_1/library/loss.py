import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.model import masked_mae_loss


class DeepSupervisionMAELoss(nn.Module):
    """
    Computes the weighted sum of Mean Absolute Error (MAE) losses for deep supervision.

    This loss function handles multiple model outputs (final prediction + auxiliary heads),
    aligns their temporal dimensions to the target if necessary, and computes a weighted
    sum of the masked MAE for each head.
    """

    def __init__(self, weights=None):
        """
        Initialize the DeepSupervisionMAELoss.

        Args:
            weights (list of float, optional): A list of weights corresponding to the model outputs.
                                               Order should be [final_head, aux_head_1, aux_head_2, ...].
                                               Defaults to Config.LOSS_WEIGHTS.
        """
        super(DeepSupervisionMAELoss, self).__init__()
        self.weights = weights if weights is not None else Config.LOSS_WEIGHTS

    def forward(self, preds, target, mask):
        """
        Compute the weighted masked MAE loss.

        Args:
            preds (list of torch.Tensor): List of model predictions. The first element is treated
                                          as the final output, and subsequent elements as auxiliary outputs.
                                          Each tensor has shape (Batch, Channels, Length).
            target (torch.Tensor): Ground truth targets of shape (Batch, Channels, Length).
            mask (torch.Tensor): Boolean or float mask of shape (Batch, Length) indicating valid time steps.
                                 1.0 denotes valid data, 0.0 denotes padding.

        Returns:
            torch.Tensor: The scalar weighted total loss.
        """
        total_loss = 0.0

        # Iterate over predictions and corresponding weights
        # We use zip to pair them; if lengths differ, it stops at the shorter one.
        for i, (pred, weight) in enumerate(zip(preds, self.weights)):

            # Align temporal dimension (Length) if necessary
            # pred shape: (B, C, L_pred), target shape: (B, C, L_target)
            if pred.shape[2] != target.shape[2]:
                pred = F.interpolate(
                    pred, size=target.shape[2], mode="linear", align_corners=True
                )

            # Compute masked MAE for the current head
            # masked_mae_loss handles the masking internally
            head_loss = masked_mae_loss(pred, target, mask)

            # Add weighted contribution to total loss
            total_loss += weight * head_loss

        return total_loss
