import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class BCEDiceLoss(nn.Module):
    """
    Loss function combining Binary Cross Entropy and Dice Loss.
    Designed for the 2.5D BiSeNet architecture (Idea 9).

    Features:
    - Differentiable Dice Loss implementation.
    - Automatic handling of BiSeNet's dual outputs (Main + Auxiliary).
    - Configurable weighting for Aux loss.
    """

    def __init__(self, bce_weight=0.5, smooth=1e-6):
        """
        Args:
            bce_weight (float): Weight for BCE component (1 - bce_weight for Dice).
            smooth (float): Smoothing factor for Dice calculation to avoid div by zero.
        """
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.aux_weight = Config.AUX_LOSS_WEIGHT
        self.bce_func = nn.BCEWithLogitsLoss()

    def _dice_loss(self, pred_logits, targets):
        """
        Computes the Dice Loss in a differentiable manner.

        Args:
            pred_logits: Raw output from the model (before sigmoid).
            targets: Ground truth binary masks.

        Returns:
            torch.Tensor: Scalar Dice loss (1 - Dice Coefficient).
        """
        # Apply Sigmoid to get probabilities
        pred_probs = torch.sigmoid(pred_logits)

        # Calculate intersection and union
        # Summing over spatial dimensions (H, W) -> dims (2, 3)
        # We keep Batch and Channel dimensions to average later
        intersection = (pred_probs * targets).sum(dim=(2, 3))
        union = pred_probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        # Dice Coefficient
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return 1 - Dice (averaged over batch and channels)
        return 1.0 - dice_score.mean()

    def _compute_single_loss(self, pred, target):
        """
        Computes the weighted sum of BCE and Dice for a single prediction tensor.
        """
        bce = self.bce_func(pred, target)
        dice = self._dice_loss(pred, target)

        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice

    def forward(self, preds, targets):
        """
        Calculate loss. Handles both single tensor output and BiSeNet tuple output.

        Args:
            preds: Model output. Can be:
                   - Tensor of shape (B, C, H, W)
                   - Tuple of (Main_Output, Aux_Output)
            targets: Ground truth of shape (B, C, H, W)

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Check for BiSeNet tuple output (Main, Aux)
        if isinstance(preds, (tuple, list)):
            main_pred, aux_pred = preds

            # Calculate losses for both heads
            loss_main = self._compute_single_loss(main_pred, targets)
            loss_aux = self._compute_single_loss(aux_pred, targets)

            # Combine: Primary + 0.1 * Auxiliary
            return loss_main + self.aux_weight * loss_aux

        else:
            # Standard single output case
            return self._compute_single_loss(preds, targets)
