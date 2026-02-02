import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceLoss(nn.Module):
    """
    Dice Coefficient Loss for binary segmentation.

    Formula:
        Loss = 1 - (2 * Intersection + smooth) / (Union + smooth)

    Expects logits as input (applies sigmoid internally).
    """

    def __init__(self, smooth=1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output of shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth of shape (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_score


class DeepSupervisionLoss(nn.Module):
    """
    Composite loss function for Deep Supervision in U-Net++.

    Combines BCEWithLogitsLoss and DiceLoss.
    Handles both single tensor output and list of tensors (deep supervision).
    """

    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

        # Load weights from config or default to equal weighting if not specified
        if (
            hasattr(Config, "DEEP_SUPERVISION_WEIGHTS")
            and Config.DEEP_SUPERVISION_WEIGHTS
        ):
            self.weights = Config.DEEP_SUPERVISION_WEIGHTS
        else:
            self.weights = [1.0]

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor or List[torch.Tensor]):
                - If Deep Supervision is ON: List of logits from decoder levels.
                - If Deep Supervision is OFF: Single tensor of logits.
            targets (torch.Tensor): Ground truth mask (B, 1, H, W).

        Returns:
            torch.Tensor: Weighted sum of losses.
        """
        # Handle single output case (wrap in list)
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        loss = 0.0

        # Iterate over predictions and corresponding weights
        # If model outputs more levels than we have weights for, we ignore extra deep levels
        # If model outputs fewer, we just use what we have.
        for i, pred in enumerate(preds):
            if i < len(self.weights):
                weight = self.weights[i]

                # Calculate individual losses
                bce_loss = self.bce(pred, targets)
                dice_loss = self.dice(pred, targets)

                # Combine (0.5 BCE + 0.5 Dice is a common strategy,
                # but here we sum them as they are on similar scales usually,
                # or we can treat them as 1:1)
                # Standard practice: Loss = BCE + Dice
                stage_loss = bce_loss + dice_loss

                loss += weight * stage_loss

        return loss
