import torch
import torch.nn as nn
from library.config import cfg


class CompositeLoss(nn.Module):
    """
    Composite Loss function for the ResNet18-D Multi-Task U-Net.
    Combines CrossEntropyLoss for study-level classification and
    BCEWithLogitsLoss for image-level segmentation.
    """

    def __init__(self):
        super(CompositeLoss, self).__init__()

        # Study-level loss: Multi-class classification (Negative, Typical, Indeterminate, Atypical)
        # We use CrossEntropyLoss which expects raw logits and integer class indices.
        self.study_criterion = nn.CrossEntropyLoss()

        # Image-level loss: Binary segmentation (Opacity vs Background)
        # We use BCEWithLogitsLoss which is more numerically stable than sigmoid + BCE.
        self.image_criterion = nn.BCEWithLogitsLoss()

        # Loss weights from config
        self.study_weight = cfg.study_loss_weight
        self.image_weight = cfg.image_loss_weight

    def forward(self, cls_logits, seg_logits, cls_targets, seg_targets):
        """
        Computes the weighted composite loss.

        Args:
            cls_logits (torch.Tensor): Study predictions of shape (B, num_study_classes).
            seg_logits (torch.Tensor): Segmentation predictions of shape (B, 1, H, W).
            cls_targets (torch.Tensor): Ground truth study labels of shape (B,).
            seg_targets (torch.Tensor): Ground truth masks of shape (B, 1, H, W).

        Returns:
            tuple: (total_loss, study_loss, image_loss)
        """
        # 1. Compute Study Loss
        # cls_targets should be LongTensor of class indices
        loss_cls = self.study_criterion(cls_logits, cls_targets)

        # 2. Compute Image Loss
        # seg_targets should be FloatTensor matching seg_logits shape
        loss_seg = self.image_criterion(seg_logits, seg_targets)

        # 3. Weighted Combination
        total_loss = (self.study_weight * loss_cls) + (self.image_weight * loss_seg)

        return total_loss, loss_cls, loss_seg
