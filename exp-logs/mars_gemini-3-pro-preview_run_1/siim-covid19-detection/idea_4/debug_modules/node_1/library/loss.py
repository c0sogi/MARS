import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceLoss(nn.Module):
    """
    Dice Coefficient Loss for binary segmentation.
    Computes 1 - Dice Score.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class HybridLoss(nn.Module):
    """
    Hybrid Loss function combining Study-level CrossEntropy and
    Image-level BCE + Dice Loss with Deep Supervision support.
    """

    def __init__(self):
        super(HybridLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

        self.cls_weight = Config.LOSS_WEIGHT_CLS
        self.seg_weight = Config.LOSS_WEIGHT_SEG
        self.deep_supervision = Config.DEEP_SUPERVISION

    def forward(self, study_logits, mask_logits, study_targets, mask_targets):
        """
        Args:
            study_logits (torch.Tensor): (B, 4) Raw logits for study classification.
            mask_logits (torch.Tensor or list): (B, 1, H, W) or list of tensors for deep supervision.
            study_targets (torch.Tensor): (B, 4) One-hot encoded targets or (B,) indices.
            mask_targets (torch.Tensor): (B, 1, H, W) Binary segmentation masks.

        Returns:
            dict: Dictionary containing 'loss' (total) and components for logging.
        """

        # ===========================
        # 1. Study-Level Loss
        # ===========================
        # Ensure targets are class indices for CrossEntropyLoss
        if study_targets.dim() == 2 and study_targets.shape[1] == Config.NUM_CLASSES:
            # Convert one-hot to indices
            study_targets_indices = torch.argmax(study_targets, dim=1)
        else:
            study_targets_indices = study_targets.long()

        study_loss = self.ce_loss(study_logits, study_targets_indices)

        # ===========================
        # 2. Image-Level Loss
        # ===========================
        seg_loss = 0.0
        bce_loss_val = 0.0
        dice_loss_val = 0.0

        # Ensure mask targets are float for BCE/Dice
        mask_targets = mask_targets.float()

        if self.deep_supervision and isinstance(mask_logits, (list, tuple)):
            # Deep Supervision: mask_logits is [output_final, output_half, output_quarter, ...]
            # Standard weights for U-Net deep supervision
            weights = [1.0, 0.5, 0.25, 0.125]

            for i, pred in enumerate(mask_logits):
                # Use default small weight if we exceed defined weights
                weight = weights[i] if i < len(weights) else 0.1

                # Resize target to match prediction resolution
                if pred.shape[-2:] != mask_targets.shape[-2:]:
                    target_resized = F.interpolate(
                        mask_targets, size=pred.shape[-2:], mode="nearest"
                    )
                else:
                    target_resized = mask_targets

                bce = self.bce_loss(pred, target_resized)
                dice = self.dice_loss(pred, target_resized)

                # Sum components
                loss_component = bce + dice
                seg_loss += weight * loss_component

                # Log metrics for the primary head (full resolution)
                if i == 0:
                    bce_loss_val = bce
                    dice_loss_val = dice
        else:
            # Single Output
            # Handle case where deep supervision is off but model returns a list (e.g. just one element)
            if isinstance(mask_logits, (list, tuple)):
                mask_logits = mask_logits[0]

            bce_loss_val = self.bce_loss(mask_logits, mask_targets)
            dice_loss_val = self.dice_loss(mask_logits, mask_targets)
            seg_loss = bce_loss_val + dice_loss_val

        # ===========================
        # 3. Total Loss Aggregation
        # ===========================
        # Apply strict 1:10 weighting ratio as per strategy
        total_loss = (self.cls_weight * study_loss) + (self.seg_weight * seg_loss)

        return {
            "loss": total_loss,
            "study_loss": study_loss,
            "seg_loss": seg_loss,
            "bce_loss": bce_loss_val,
            "dice_loss": dice_loss_val,
        }
