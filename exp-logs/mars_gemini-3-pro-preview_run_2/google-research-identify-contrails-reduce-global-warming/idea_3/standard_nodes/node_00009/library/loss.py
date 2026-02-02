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


class MultiTaskLoss(nn.Module):
    """
    Joint Loss for Multi-Task Learning:
    1. Segmentation: Dice Loss + BCE Loss
    2. Classification: BCE Loss
    Total Loss = L_seg + lambda * L_cls
    """

    def __init__(self):
        super(MultiTaskLoss, self).__init__()

        # Sub-losses
        self.dice_loss = DiceLoss()
        self.bce_seg_loss = nn.BCEWithLogitsLoss()
        self.bce_cls_loss = nn.BCEWithLogitsLoss()

        # Weight for the auxiliary classification task
        self.cls_weight = Config.CLS_WEIGHT

    def forward(self, seg_logits, cls_logits, mask_targets, cls_targets):
        """
        Args:
            seg_logits (torch.Tensor): (B, 1, H, W) Segmentation head output.
            cls_logits (torch.Tensor): (B, 1) Classification head output.
            mask_targets (torch.Tensor): (B, 1, H, W) Ground truth masks.
            cls_targets (torch.Tensor): (B,) or (B, 1) Ground truth class labels.

        Returns:
            dict: Dictionary containing 'loss' (total loss for backprop) and components.
        """
        # Ensure targets are float for BCE/Dice calculations
        if not mask_targets.is_floating_point():
            mask_targets = mask_targets.float()
        if not cls_targets.is_floating_point():
            cls_targets = cls_targets.float()

        # ---------------------------------------------------
        # 1. Segmentation Loss (Dice + BCE)
        # ---------------------------------------------------
        loss_dice = self.dice_loss(seg_logits, mask_targets)
        loss_bce_seg = self.bce_seg_loss(seg_logits, mask_targets)

        loss_seg = loss_dice + loss_bce_seg

        # ---------------------------------------------------
        # 2. Classification Loss (BCE)
        # ---------------------------------------------------
        # Ensure cls_targets matches logits shape (B, 1)
        if cls_targets.ndim == 1:
            cls_targets = cls_targets.view(-1, 1)

        loss_cls = self.bce_cls_loss(cls_logits, cls_targets)

        # ---------------------------------------------------
        # 3. Total Loss
        # ---------------------------------------------------
        total_loss = loss_seg + (self.cls_weight * loss_cls)

        return {
            "loss": total_loss,
            "seg_loss": loss_seg,
            "cls_loss": loss_cls,
            "dice_loss": loss_dice,
            "bce_seg_loss": loss_bce_seg,
        }
