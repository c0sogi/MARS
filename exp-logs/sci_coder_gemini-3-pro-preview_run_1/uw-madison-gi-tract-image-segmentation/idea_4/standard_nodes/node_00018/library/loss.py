import torch
import torch.nn as nn
from library.config import Config


class DeepSupervisionLoss(nn.Module):
    """
    Implements a combined Binary Cross Entropy (BCE) and Dice Loss,
    specifically designed to handle Deep Supervision outputs from U-Net++.
    """

    def __init__(self, bce_weight=None, dice_weight=None):
        """
        Args:
            bce_weight (float, optional): Weight for BCE loss. Defaults to Config.BCE_WEIGHT.
            dice_weight (float, optional): Weight for Dice loss. Defaults to Config.DICE_WEIGHT.
        """
        super(DeepSupervisionLoss, self).__init__()
        self.bce_weight = bce_weight if bce_weight is not None else Config.BCE_WEIGHT
        self.dice_weight = (
            dice_weight if dice_weight is not None else Config.DICE_WEIGHT
        )

        # BCEWithLogitsLoss combines Sigmoid layer and the BCELoss in one single class.
        # This is more numerically stable than using a plain Sigmoid followed by a BCELoss.
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = 1e-7

    def forward(self, y_pred, y_true):
        """
        Calculates the loss. Handles both single tensor output and list/tuple outputs
        (for Deep Supervision).

        Args:
            y_pred (torch.Tensor or list/tuple): Predicted logits.
                Shape (B, C, H, W) or list of such tensors.
            y_true (torch.Tensor): Ground truth masks. Shape (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Check for Deep Supervision (list of outputs from intermediate decoder layers)
        if isinstance(y_pred, (list, tuple)):
            loss = 0.0
            # Use getattr with default to handle potential stale Config in persistent runtimes (Cite debug_lesson_6)
            weights = getattr(Config, "DS_WEIGHTS", [0.1, 0.1, 0.1, 0.7])

            # Fallback if weights config doesn't match output length
            if len(weights) != len(y_pred):
                weights = [1.0] * len(y_pred)

            total_weight = sum(weights)

            for i, pred in enumerate(y_pred):
                # Apply specific weight to each head
                loss += weights[i] * self._compute_combined_loss(pred, y_true)

            # Normalize by total weight to maintain gradient scale
            return loss / total_weight
        else:
            return self._compute_combined_loss(y_pred, y_true)

    def _compute_combined_loss(self, pred, target):
        """
        Computes the weighted sum of BCE and Dice loss for a single prediction tensor.
        """
        # 1. Binary Cross Entropy Loss
        bce_loss = self.bce(pred, target)

        # 2. Dice Loss
        # pred contains logits, so apply sigmoid for Dice calculation
        pred_prob = torch.sigmoid(pred)

        # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
        # We calculate Dice per sample (B) and per class (C), then average.
        batch_size = pred_prob.size(0)
        num_classes = pred_prob.size(1)

        pred_flat = pred_prob.view(batch_size, num_classes, -1)
        target_flat = target.view(batch_size, num_classes, -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        # Dice coefficient per channel per sample
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Mean Dice Loss (1 - Dice)
        dice_loss = 1.0 - dice_score.mean()

        # Weighted Combination
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
