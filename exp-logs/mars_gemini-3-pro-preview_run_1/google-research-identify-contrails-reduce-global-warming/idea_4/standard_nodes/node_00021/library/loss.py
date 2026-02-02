import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceLoss(nn.Module):
    """
    Implements the Dice Loss for binary segmentation.
    Loss = 1 - Dice Coefficient
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model predictions (before sigmoid), shape (B, 1, H, W) or (B, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (B, 1, H, W) or (B, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to ensure computation over the entire volume/image
        # This handles both (B, 1, H, W) and (B, H, W) shapes
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Calculate intersection and union
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Compute Dice Coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice Loss
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Implements Binary Focal Loss.
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
            logits (torch.Tensor): Raw model predictions (before sigmoid).
            targets (torch.Tensor): Ground truth binary masks.

        Returns:
            torch.Tensor: Loss value.
        """
        # Ensure targets are float for BCE calculation
        targets = targets.float()

        # Compute binary cross entropy loss (element-wise)
        # BCEWithLogitsLoss is more numerically stable than Sigmoid + BCELoss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Calculate probabilities (p_t)
        pt = torch.exp(-bce_loss)

        # Calculate alpha weighting
        # If target=1, weight is alpha. If target=0, weight is (1-alpha).
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ContrailLoss(nn.Module):
    """
    Composite loss function for Contrail Segmentation.
    Combines Focal Loss and Dice Loss.
    Total Loss = Focal Loss + Dice Loss
    """

    def __init__(self, alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA, smooth=1e-6):
        super(ContrailLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw model predictions (B, C, H, W).
            targets (torch.Tensor): Ground truth masks (B, C, H, W) or (B, H, W).

        Returns:
            torch.Tensor: Combined loss.
        """
        # Ensure targets match logits shape if necessary (e.g. add channel dim)
        if logits.dim() == 4 and targets.dim() == 3:
            targets = targets.unsqueeze(1)

        # Ensure inputs are on the same device
        if targets.device != logits.device:
            targets = targets.to(logits.device)

        loss_focal = self.focal_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)

        return loss_focal + loss_dice
