import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """
    Loss function combining Binary Cross Entropy (BCE) and Dice Loss.

    This combination is effective for segmentation tasks with class imbalance,
    such as ink detection where ink pixels are sparse compared to background.
    BCE provides pixel-wise supervision, while Dice Loss optimizes for the
    overlap metric (F1 score surrogate).
    """

    def __init__(self, bce_weight=0.5, smooth=1e-5):
        """
        Args:
            bce_weight (float): Weight assigned to BCE loss. Dice loss gets (1 - bce_weight).
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
        """
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Ensure targets are float for calculation
        targets = targets.float()

        # --- BCE Loss ---
        # binary_cross_entropy_with_logits includes sigmoid for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")

        # --- Dice Loss ---
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(inputs)

        # Flatten the tensors to compute Dice over the entire batch or image
        # Using view(-1) flattens all dimensions
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # --- Combined Loss ---
        loss = (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * dice_loss)

        return loss
