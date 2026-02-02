import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    """
    Loss function combining Binary Cross Entropy and Dice Loss.
    Used for the Coarse Stage (Stage 1) to balance pixel-wise accuracy and overlap.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (N, C, H, W).
        """
        # Ensure targets are float
        targets = targets.float()

        # BCE Loss (numerically stable with logits)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)

        # Dice Loss
        inputs_prob = torch.sigmoid(inputs)

        # Flatten tensors for global Dice calculation
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            inputs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss for handling class imbalance and prioritizing recall.
    Used for the Fine Stage (Stage 2).

    Tversky Index = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    Loss = 1 - Tversky Index

    Args:
        alpha (float): Weight for False Positives.
        beta (float): Weight for False Negatives. Set beta > alpha to prioritize recall.
        smooth (float): Smoothing factor to avoid division by zero.
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits) of shape (N, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (N, C, H, W).
        """
        # Ensure targets are float
        targets = targets.float()

        inputs_prob = torch.sigmoid(inputs)

        # Flatten tensors
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)

        # Calculate True Positives, False Positives, False Negatives
        tp = (inputs_flat * targets_flat).sum()
        fp = ((1.0 - targets_flat) * inputs_flat).sum()
        fn = (targets_flat * (1.0 - inputs_flat)).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return 1.0 - tversky_index
