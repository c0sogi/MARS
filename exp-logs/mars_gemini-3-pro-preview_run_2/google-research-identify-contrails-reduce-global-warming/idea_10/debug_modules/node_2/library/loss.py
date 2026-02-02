import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchDiceLoss(nn.Module):
    """
    Computes the Dice Loss over the entire batch (flattened).
    This is useful for sparse segmentation tasks where individual images might
    have no positive pixels, leading to unstable gradients if averaged per image.
    """

    def __init__(self, smooth=1e-6):
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs (before sigmoid), shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth masks, shape (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value (1 - Dice).
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten inputs to (N,) to compute global batch statistics
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice_score


class HybridLoss(nn.Module):
    """
    Combines Binary Cross Entropy (BCE) and Batch Dice Loss.
    BCE provides smooth gradients for pixel-wise classification.
    Dice directly optimizes the IoU-like metric.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super(HybridLoss, self).__init__()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.dice_loss_fn = BatchDiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs, shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth masks, shape (B, 1, H, W).

        Returns:
            torch.Tensor: Weighted combined loss.
        """
        bce = self.bce_loss_fn(logits, targets)
        dice = self.dice_loss_fn(logits, targets)

        return (self.bce_weight * bce) + (self.dice_weight * dice)


class DeepSupervisionLoss(nn.Module):
    """
    Computes the total loss for the Cascaded ResNet18 U-Net architecture.
    Aggregates HybridLoss from both Stage 1 (Detector) and Stage 2 (Refiner).
    """

    def __init__(
        self,
        stage1_weight=1.0,
        stage2_weight=1.0,
        bce_weight=1.0,
        dice_weight=1.0,
        smooth=1e-6,
    ):
        super(DeepSupervisionLoss, self).__init__()
        self.hybrid_loss = HybridLoss(
            bce_weight=bce_weight, dice_weight=dice_weight, smooth=smooth
        )
        self.stage1_weight = stage1_weight
        self.stage2_weight = stage2_weight

    def forward(self, outputs, targets):
        """
        Args:
            outputs (tuple): A tuple containing (stage1_logits, stage2_logits).
                             Each tensor has shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth masks, shape (B, 1, H, W).

        Returns:
            torch.Tensor: Total weighted loss.
            dict: Dictionary containing individual loss components for logging.
        """
        stage1_logits, stage2_logits = outputs

        # Ensure targets match logits shape/type if necessary
        # Usually targets are float for BCEWithLogitsLoss
        if targets.dtype != stage1_logits.dtype:
            targets = targets.type_as(stage1_logits)

        loss_stage1 = self.hybrid_loss(stage1_logits, targets)
        loss_stage2 = self.hybrid_loss(stage2_logits, targets)

        total_loss = (self.stage1_weight * loss_stage1) + (
            self.stage2_weight * loss_stage2
        )

        metrics = {
            "loss_total": total_loss.item(),
            "loss_stage1": loss_stage1.item(),
            "loss_stage2": loss_stage2.item(),
        }

        return total_loss, metrics
