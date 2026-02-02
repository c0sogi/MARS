import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceLoss(nn.Module):
    """
    Dice Coefficient Loss for binary segmentation.
    Computes 1 - DiceScore.
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw output from the model (N, 1, H, W).
            targets (torch.Tensor): Ground truth masks (N, 1, H, W).
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw output from the model (N, 1, H, W).
            targets (torch.Tensor): Ground truth masks (N, 1, H, W).
        """
        # Compute binary cross entropy loss
        # reduction='none' to keep per-pixel loss for weighting
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Calculate p_t
        # p_t = p if y=1 else 1-p
        # BCEWithLogitsLoss is -log(p_t), so p_t = exp(-bce_loss)
        p_t = torch.exp(-bce_loss)

        # Calculate alpha_t
        # alpha_t = alpha if y=1 else (1-alpha)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate Focal Loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of Focal Loss and Dice Loss.
    Loss = w_focal * FocalLoss + w_dice * DiceLoss
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()

        # Initialize components using Config
        self.focal_loss = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
        self.dice_loss = DiceLoss()

        # Weights
        self.focal_weight = Config.FOCAL_LOSS_WEIGHT
        self.dice_weight = Config.DICE_LOSS_WEIGHT

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw output from the model (N, 1, H, W).
            targets (torch.Tensor): Ground truth masks (N, 1, H, W).
        """
        f_loss = self.focal_loss(logits, targets)
        d_loss = self.dice_loss(logits, targets)

        total_loss = (self.focal_weight * f_loss) + (self.dice_weight * d_loss)

        return total_loss
