import torch
import torch.nn as nn
from library.config import Config


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.
    Formula: 1 - (2 * |X n Y| + smooth) / (|X| + |Y| + smooth)
    Applied on sigmoid probabilities.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): (B, 1, H, W) raw model outputs (before sigmoid).
            targets (torch.Tensor): (B, 1, H, W) binary ground truth masks.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: (B, 1, H, W) -> (B, H*W)
        # We calculate Dice per image in the batch, then average.
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        cardinality = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Loss is 1 - Dice
        return 1.0 - dice_score.mean()


class SegmentationLoss(nn.Module):
    """
    Segmentation Loss: Dice Loss + BCE Loss
    """

    def __init__(self):
        super(SegmentationLoss, self).__init__()

        # Sub-losses
        self.dice_loss = DiceLoss()
        self.bce_seg_loss = nn.BCEWithLogitsLoss()

    def forward(self, seg_logits, mask_targets):
        """
        Args:
            seg_logits (torch.Tensor): (B, 1, H, W) Segmentation head output.
            mask_targets (torch.Tensor): (B, 1, H, W) Ground truth masks.

        Returns:
            dict: Dictionary containing 'loss' (total loss for backprop) and components.
        """
        # Ensure targets are float for BCE/Dice calculations
        if not mask_targets.is_floating_point():
            mask_targets = mask_targets.float()

        # ---------------------------------------------------
        # 1. Segmentation Loss (Dice + BCE)
        # ---------------------------------------------------
        loss_dice = self.dice_loss(seg_logits, mask_targets)
        loss_bce_seg = self.bce_seg_loss(seg_logits, mask_targets)

        loss_seg = loss_dice + loss_bce_seg

        return {
            "loss": loss_seg,
            "dice_loss": loss_dice,
            "bce_seg_loss": loss_bce_seg,
        }
