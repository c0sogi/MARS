import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceLoss(nn.Module):
    """
    Implements the Soft Dice Loss for binary segmentation.
    Formula: 1 - (2 * |X n Y| + smooth) / (|X| + |Y| + smooth)
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs of shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, 1, H, W).
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to compute global stats
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Implements Binary Focal Loss.
    Formula: -alpha * (1 - pt)^gamma * log(pt)
    """

    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model outputs of shape (B, 1, H, W).
            targets (torch.Tensor): Ground truth binary masks of shape (B, 1, H, W).
        """
        # Compute binary cross entropy with logits (numerically stable)
        # reduction='none' to apply weighting element-wise
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Calculate probabilities
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        # Calculate alpha weighting
        # If target=1, weight is alpha. If target=0, weight is 1-alpha.
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Calculate focal term
        focal_term = (1 - pt) ** self.gamma

        # Combine terms
        loss = alpha_t * focal_term * bce_loss

        return loss.mean()


class ContrailLoss(nn.Module):
    """
    Composite loss function combining Focal Loss and Dice Loss.
    Strategy: Balanced Imbalance-Aware Optimization.
    """

    def __init__(self):
        super(ContrailLoss, self).__init__()

        # Initialize components with configuration parameters
        self.focal_loss = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
        self.dice_loss = DiceLoss()

        # Weights for the composite loss
        self.focal_weight = Config.FOCAL_WEIGHT
        self.dice_weight = Config.DICE_WEIGHT

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output logits (B, 1, H, W).
            targets (torch.Tensor): Ground truth masks (B, 1, H, W).
        """
        # Ensure targets are float for calculation
        if targets.dtype != torch.float32:
            targets = targets.float()

        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total_loss = (self.focal_weight * focal) + (self.dice_weight * dice)

        return total_loss
