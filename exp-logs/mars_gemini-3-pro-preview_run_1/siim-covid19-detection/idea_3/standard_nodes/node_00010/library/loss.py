import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Calculates the Sørensen–Dice coefficient loss for binary segmentation.
    Expects logits as input (applies Sigmoid internally).
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output logits of shape (N, C, H, W) or (N, H, W).
            targets (torch.Tensor): Ground truth masks of shape (N, C, H, W) or (N, H, W).

        Returns:
            torch.Tensor: Scalar loss value (1 - Dice Score).
        """
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


class HybridLoss(nn.Module):
    """
    Combines study-level classification loss and image-level segmentation loss.

    L_total = w_cls * L_cls + w_seg * (L_bce_seg + L_dice_seg)
    """

    def __init__(self, seg_weight=1.0, cls_weight=1.0):
        super(HybridLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

        # Study-level classification loss (Multi-label/Multi-class)
        # We use BCEWithLogitsLoss as the targets are one-hot encoded floats
        self.cls_criterion = nn.BCEWithLogitsLoss()

        # Segmentation losses
        self.seg_bce = nn.BCEWithLogitsLoss()
        self.seg_dice = DiceLoss()

    def forward(self, cls_logits, seg_logits, cls_targets, seg_targets):
        """
        Args:
            cls_logits (torch.Tensor): Classification logits (B, NumClasses).
            seg_logits (torch.Tensor): Segmentation logits (B, 1, H, W).
            cls_targets (torch.Tensor): Classification targets (B, NumClasses).
            seg_targets (torch.Tensor): Segmentation targets (B, 1, H, W).

        Returns:
            tuple: (total_loss, cls_loss, seg_loss, dice_loss_val)
        """
        # 1. Classification Loss
        # Ensure targets are float for BCE
        cls_loss = self.cls_criterion(cls_logits, cls_targets.float())

        # 2. Segmentation Loss
        # Ensure targets are float
        seg_targets = seg_targets.float()

        # Pixel-wise Binary Cross Entropy
        bce_seg_loss = self.seg_bce(seg_logits, seg_targets)

        # Dice Loss
        dice_seg_loss = self.seg_dice(seg_logits, seg_targets)

        # Combined Segmentation Loss
        total_seg_loss = bce_seg_loss + dice_seg_loss

        # 3. Total Hybrid Loss
        total_loss = (self.cls_weight * cls_loss) + (self.seg_weight * total_seg_loss)

        return total_loss, cls_loss, total_seg_loss, dice_seg_loss
