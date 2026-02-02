import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class MultiTaskLoss(nn.Module):
    """
    Combined loss function for Multi-Task Learning.
    Combines CrossEntropyLoss for classification and BCEWithLogitsLoss for segmentation.
    Applies weights defined in Config.
    """

    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.seg_criterion = nn.BCEWithLogitsLoss()

        self.cls_weight = Config.LOSS_WEIGHTS["class"]
        self.seg_weight = Config.LOSS_WEIGHTS["seg"]

    def forward(self, cls_logits, seg_logits, cls_targets, seg_targets):
        """
        Args:
            cls_logits (torch.Tensor): Classification logits (N, NumClasses).
            seg_logits (torch.Tensor): Segmentation logits (N, 1, H, W).
            cls_targets (torch.Tensor): Classification one-hot targets (N, NumClasses).
            seg_targets (torch.Tensor): Segmentation masks (N, 1, H, W).

        Returns:
            dict: Dictionary containing 'loss' (total weighted loss),
                  'cls_loss' (unweighted), and 'seg_loss' (unweighted).
        """
        # Study-Level Loss (Classification)
        # Convert one-hot to indices for CrossEntropyLoss
        cls_targets_idx = torch.argmax(cls_targets, dim=1)
        cls_loss = self.cls_criterion(cls_logits, cls_targets_idx)

        # Image-Level Loss (Segmentation)
        # Ensure targets are float for BCE
        seg_loss = self.seg_criterion(seg_logits, seg_targets.float())

        # Weighted Sum
        total_loss = (self.cls_weight * cls_loss) + (self.seg_weight * seg_loss)

        return {"loss": total_loss, "cls_loss": cls_loss, "seg_loss": seg_loss}
