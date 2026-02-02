import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceLoss(nn.Module):
    """
    Dice Coefficient Loss for binary segmentation.
    Loss = 1 - (2 * Intersection + Smooth) / (Union + Smooth)
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) Raw model output (before sigmoid).
            targets: (B, 1, H, W) Binary ground truth masks.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Calculate Dice score
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1.0 - dice_score


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, reduction="mean"
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) Raw model output.
            targets: (B, 1, H, W) Binary ground truth masks.
        """
        # Calculate Binary Cross Entropy with Logits
        # reduction='none' to apply focal weights per pixel
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Calculate probabilities (p_t)
        # pt is the probability of the true class
        # If target=1, pt = sigmoid(logits)
        # If target=0, pt = 1 - sigmoid(logits)
        # Fortunately, BCEWithLogitsLoss gives us -log(pt), so pt = exp(-bce_loss)
        pt = torch.exp(-bce_loss)

        # Calculate Focal Term: (1 - pt)^gamma
        focal_term = (1.0 - pt) ** self.gamma

        # Calculate Alpha Term
        # alpha applies to class 1, (1-alpha) applies to class 0
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_term * bce_loss
        else:
            loss = focal_term * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CompositeLoss(nn.Module):
    """
    Weighted combination of Focal Loss and Dice Loss.
    Loss = weight_focal * FocalLoss + weight_dice * DiceLoss
    """

    def __init__(
        self,
        weight_focal=Config.WEIGHT_FOCAL,
        weight_dice=Config.WEIGHT_DICE,
        focal_alpha=Config.FOCAL_ALPHA,
        focal_gamma=Config.FOCAL_GAMMA,
        dice_smooth=1.0,
    ):
        super(CompositeLoss, self).__init__()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) Raw model output.
            targets: (B, 1, H, W) Binary ground truth masks.
        """
        # Ensure targets are float for calculation
        targets = targets.float()

        loss_f = self.focal_loss(logits, targets)
        loss_d = self.dice_loss(logits, targets)

        return self.weight_focal * loss_f + self.weight_dice * loss_d
