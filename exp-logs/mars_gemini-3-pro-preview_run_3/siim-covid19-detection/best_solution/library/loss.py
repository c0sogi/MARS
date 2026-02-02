import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.
    Calculates 1 - Dice Coefficient to be minimized.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model of shape (B, 1, H, W).
            targets (torch.Tensor): Binary ground truth masks of shape (B, 1, H, W).
        """
        # Apply sigmoid to convert logits to probabilities
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


class MultiTaskLoss(nn.Module):
    """
    Composite loss function for the Multi-Task U-Net.
    Combines DiceLoss (for segmentation) and CrossEntropyLoss (for classification).
    """

    def __init__(self, seg_weight=1.0, class_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.seg_weight = seg_weight
        self.class_weight = class_weight

        self.dice_loss = DiceLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, seg_logits, class_logits, mask_targets, class_targets):
        """
        Args:
            seg_logits (torch.Tensor): Segmentation logits (B, 1, H, W).
            class_logits (torch.Tensor): Classification logits (B, NumClasses).
            mask_targets (torch.Tensor): Ground truth masks (B, 1, H, W).
            class_targets (torch.Tensor): One-hot encoded ground truth labels (B, NumClasses).

        Returns:
            total_loss (torch.Tensor): The weighted sum of segmentation and classification losses.
            metrics (dict): A dictionary containing individual loss components for logging.
        """
        # 1. Calculate Segmentation Loss
        loss_seg = self.dice_loss(seg_logits, mask_targets)

        # 2. Calculate Classification Loss
        # The dataset returns one-hot encoded targets (float32).
        # CrossEntropyLoss expects class indices (long).
        target_indices = torch.argmax(class_targets, dim=1)
        loss_class = self.ce_loss(class_logits, target_indices)

        # 3. Weighted Sum
        total_loss = (self.seg_weight * loss_seg) + (self.class_weight * loss_class)

        return total_loss, {
            "seg_loss": loss_seg.item(),
            "class_loss": loss_class.item(),
            "total_loss": total_loss.item(),
        }
