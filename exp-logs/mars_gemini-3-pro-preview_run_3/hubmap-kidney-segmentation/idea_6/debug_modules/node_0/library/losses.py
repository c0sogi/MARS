import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross Entropy Loss.
    Useful for segmentation tasks to address class imbalance and optimize for the Dice metric.
    """

    def __init__(self, smooth=1e-6, bce_weight=0.5, dice_weight=0.5):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero in Dice calculation.
            bce_weight (float): Weight assigned to the BCE component.
            dice_weight (float): Weight assigned to the Dice component.
        """
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output logits of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth masks of shape (B, C, H, W).

        Returns:
            torch.Tensor: Weighted sum of BCE and Dice loss.
        """
        # 1. Binary Cross Entropy Loss
        bce = self.bce_loss(logits, targets)

        # 2. Dice Loss
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten tensors to compute Dice over the batch or per image
        # Flattening over the whole batch treats it as one large volume
        inputs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score

        # Combine losses
        total_loss = (self.bce_weight * bce) + (self.dice_weight * dice_loss)

        return total_loss


class DeepSupervisionLoss(nn.Module):
    """
    Wrapper for calculating loss with Deep Supervision.
    Computes the weighted sum of losses for a list of multi-scale outputs.
    """

    def __init__(self, weights=Config.LOSS_WEIGHTS):
        """
        Args:
            weights (list[float]): List of weights corresponding to each output scale.
                                   Defaults to Config.LOSS_WEIGHTS.
        """
        super(DeepSupervisionLoss, self).__init__()
        self.weights = weights
        self.base_loss = DiceBCELoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs (list[torch.Tensor] or torch.Tensor): List of model outputs from deep supervision heads.
            targets (torch.Tensor): Ground truth mask.

        Returns:
            torch.Tensor: Weighted sum of losses.
        """
        # Handle case where model might return a single tensor (e.g., during inference mode or different arch)
        if not isinstance(outputs, list):
            return self.base_loss(outputs, targets)

        loss = 0.0

        # Validate lengths match to avoid silent errors
        if len(outputs) != len(self.weights):
            # Fallback: if lengths don't match, just use the first output (main output)
            # or use equal weights. Here we assume the first output is the primary one.
            # However, for U-Net++ as defined in model.py, it returns 4 outputs.
            # We strictly iterate up to the minimum length.
            pass

        for output, weight in zip(outputs, self.weights):
            # Calculate base loss for this scale
            # Note: Targets are the same for all scales because outputs are upsampled in the model
            scale_loss = self.base_loss(output, targets)
            loss += weight * scale_loss

        return loss
