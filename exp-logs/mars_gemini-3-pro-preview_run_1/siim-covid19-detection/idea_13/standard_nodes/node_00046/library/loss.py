import torch
import torch.nn as nn
from library.config import Config


class HybridLoss(nn.Module):
    """
    Hybrid objective function for the ResNet18-D U-Net model.
    Combines Study-Level Classification (CrossEntropy) and Image-Level Segmentation (BCE).

    Weights are defined in Config.LOSS_WEIGHTS:
    - Classification: 1.0
    - Segmentation: 10.0 (Prioritizing dense prediction)
    """

    def __init__(self):
        super(HybridLoss, self).__init__()
        self.cls_weight = Config.LOSS_WEIGHTS[0]
        self.seg_weight = Config.LOSS_WEIGHTS[1]

        # Study-Level: Multi-class classification
        # (Negative, Typical, Indeterminate, Atypical)
        # Expects raw logits and class indices
        self.cls_criterion = nn.CrossEntropyLoss()

        # Image-Level: Binary Segmentation
        # (Opacity vs Background)
        # Expects raw logits and float masks
        self.seg_criterion = nn.BCEWithLogitsLoss()

    def forward(self, cls_logits, seg_logits, cls_labels, seg_masks):
        """
        Computes the weighted sum of classification and segmentation losses.

        Args:
            cls_logits (torch.Tensor): Predicted study logits. Shape (B, 4).
            seg_logits (torch.Tensor): Predicted segmentation logits. Shape (B, 1, H, W).
            cls_labels (torch.Tensor): Ground truth study labels (indices). Shape (B,).
            seg_masks (torch.Tensor): Ground truth binary masks. Shape (B, 1, H, W).

        Returns:
            torch.Tensor: The scalar total loss used for backpropagation.
        """
        # 1. Study-Level Loss
        cls_loss = self.cls_criterion(cls_logits, cls_labels)

        # 2. Image-Level Loss
        # BCEWithLogitsLoss handles the sigmoid internally for stability
        seg_loss = self.seg_criterion(seg_logits, seg_masks)

        # 3. Weighted Combination
        total_loss = (self.cls_weight * cls_loss) + (self.seg_weight * seg_loss)

        return total_loss
