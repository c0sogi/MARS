import torch
import torch.nn as nn
from library.config import Config


class MultiTaskLoss(nn.Module):
    """
    Composite loss function for simultaneous Study Classification and Opacity Segmentation.
    Combines CrossEntropyLoss for classification and BCEWithLogitsLoss for segmentation.
    """

    def __init__(self, lambda_cls=Config.LAMBDA_CLS, lambda_seg=Config.LAMBDA_SEG):
        """
        Args:
            lambda_cls (float): Weight for the classification loss.
            lambda_seg (float): Weight for the segmentation loss.
        """
        super(MultiTaskLoss, self).__init__()
        self.lambda_cls = lambda_cls
        self.lambda_seg = lambda_seg

        # Study Level: Multi-class classification (Negative, Typical, Indeterminate, Atypical)
        # We use CrossEntropyLoss which expects raw logits and class indices.
        self.cls_criterion = nn.CrossEntropyLoss()

        # Image Level: Binary segmentation (Opacity vs Background)
        # We use BCEWithLogitsLoss which combines Sigmoid + BCE.
        self.seg_criterion = nn.BCEWithLogitsLoss()

    def forward(self, cls_logits, mask_logits, cls_targets, mask_targets):
        """
        Calculates the combined loss.

        Args:
            cls_logits (torch.Tensor): Predicted study logits (N, 4).
            mask_logits (torch.Tensor): Predicted segmentation logits (N, 1, H, W).
            cls_targets (torch.Tensor): Ground truth study labels (N, 4) [One-hot encoded].
            mask_targets (torch.Tensor): Ground truth segmentation masks (N, 1, H, W).

        Returns:
            total_loss (torch.Tensor): Weighted sum of losses.
            metrics (dict): Dictionary containing individual loss components for logging.
        """
        # --- 1. Classification Loss ---
        # cls_targets comes from the dataset as float32 one-hot vectors.
        # CrossEntropyLoss expects class indices (LongTensor).
        # We use argmax to convert one-hot to indices.
        cls_target_indices = torch.argmax(cls_targets, dim=1)
        cls_loss = self.cls_criterion(cls_logits, cls_target_indices)

        # --- 2. Segmentation Loss ---
        # mask_logits are raw scores, mask_targets are 0.0 or 1.0 floats.
        # Ensure shapes match exactly.
        seg_loss = self.seg_criterion(mask_logits, mask_targets)

        # --- 3. Combine ---
        total_loss = (self.lambda_cls * cls_loss) + (self.lambda_seg * seg_loss)

        metrics = {
            "loss_total": total_loss.item(),
            "loss_cls": cls_loss.item(),
            "loss_seg": seg_loss.item(),
        }

        return total_loss, metrics
