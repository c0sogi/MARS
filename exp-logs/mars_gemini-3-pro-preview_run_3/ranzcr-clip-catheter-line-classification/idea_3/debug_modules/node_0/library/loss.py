import torch
import torch.nn as nn
from library.config import Config


class CustomLoss(nn.Module):
    """
    Custom loss function for the Multi-Scale Feature Aggregation model.
    Combines Multi-label Binary Cross Entropy for classification and
    Pixel-wise Binary Cross Entropy for the auxiliary segmentation head.
    """

    def __init__(
        self, cls_weight=Config.CLS_LOSS_WEIGHT, aux_weight=Config.AUX_LOSS_WEIGHT
    ):
        """
        Args:
            cls_weight (float): Weight for the classification loss.
            aux_weight (float): Weight for the auxiliary segmentation loss.
        """
        super(CustomLoss, self).__init__()
        self.cls_weight = cls_weight
        self.aux_weight = aux_weight

        # Primary Task: Multi-label Classification
        # Using BCEWithLogitsLoss as the model outputs raw logits
        self.cls_criterion = nn.BCEWithLogitsLoss()

        # Auxiliary Task: Segmentation (Background Suppression)
        # Pixel-wise BCE is used to force the encoder to learn spatial localization
        self.aux_criterion = nn.BCEWithLogitsLoss()

    def forward(self, logits, mask_pred, targets, mask_true):
        """
        Computes the weighted sum of classification and segmentation losses.

        Args:
            logits (torch.Tensor): Predicted classification logits of shape (B, NumClasses).
            mask_pred (torch.Tensor): Predicted segmentation logits of shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth classification labels of shape (B, NumClasses).
            mask_true (torch.Tensor): Ground truth segmentation masks of shape (B, 1, H, W).

        Returns:
            torch.Tensor: The combined scalar loss.
        """
        # 1. Classification Loss
        # Ensure targets are float for BCE
        if targets.dtype != torch.float32:
            targets = targets.float()

        cls_loss = self.cls_criterion(logits, targets)

        # 2. Auxiliary Segmentation Loss
        # Ensure masks are float for BCE
        if mask_true.dtype != torch.float32:
            mask_true = mask_true.float()

        aux_loss = self.aux_criterion(mask_pred, mask_true)

        # 3. Weighted Combination
        total_loss = (self.cls_weight * cls_loss) + (self.aux_weight * aux_loss)

        return total_loss
