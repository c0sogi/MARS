import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceLoss(nn.Module):
    """
    Dice Coefficient Loss for binary segmentation.
    Formula: 1 - (2 * |X n Y| + smooth) / (|X| + |Y| + smooth)
    """

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) Raw output from the model (before Sigmoid).
            targets: (B, 1, H, W) Binary ground truth masks (0 or 1).
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors to calculate metric over the batch or image
        # Flattening (B, 1, H, W) -> (B, -1) to compute Dice per sample and mean,
        # or (N,) for global Dice. Standard practice for stability is often global or batch-mean.
        # Here we flatten all dims except batch to compute per-image Dice, then mean.
        inputs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (inputs_flat * targets_flat).sum(dim=1)
        union = inputs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Loss is 1 - Dice
        loss = 1.0 - dice_score

        return loss.mean()


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where p_t is the model's estimated probability for the target class.
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
            logits: (B, 1, H, W) Raw output from the model.
            targets: (B, 1, H, W) Binary ground truth masks.
        """
        # BCEWithLogitsLoss combines Sigmoid and BCE for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Calculate p_t: probability of the true class
        # p = sigmoid(logits)
        # if y=1, p_t = p; if y=0, p_t = 1-p
        # This is equivalent to exp(-bce_loss)
        p_t = torch.exp(-bce_loss)

        # Calculate alpha_t
        # if y=1, alpha_t = alpha; if y=0, alpha_t = 1-alpha
        # Note: If alpha is 0.5, alpha_t is always 0.5
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate Focal Loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedLoss(nn.Module):
    """
    Combined Loss function: Weight_Focal * FocalLoss + Weight_Dice * DiceLoss.
    Designed to prioritize pixel stability (Focal) while ensuring overlap optimization (Dice).
    """

    def __init__(self):
        super(WeightedLoss, self).__init__()
        self.focal = FocalLoss(alpha=Config.FOCAL_ALPHA, gamma=Config.FOCAL_GAMMA)
        self.dice = DiceLoss()

        self.w_focal = Config.WEIGHT_FOCAL
        self.w_dice = Config.WEIGHT_DICE

    def forward(self, logits, targets):
        focal_l = self.focal(logits, targets)
        dice_l = self.dice(logits, targets)

        total_loss = (self.w_focal * focal_l) + (self.w_dice * dice_l)

        return total_loss, focal_l, dice_l
