import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Coefficient Loss for binary/multi-label segmentation.
    Calculates 1 - Dice Score.

    Args:
        smooth (float): Smoothing factor to prevent division by zero.
        mode (str): 'binary', 'multiclass', or 'multilabel'.
                    For this task (3 independent masks), 'multilabel' is appropriate.
    """

    def __init__(self, smooth=1.0, mode="multilabel"):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.mode = mode

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth of shape (B, C, H, W).
        """
        # Apply activation
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors
        # Shape transformation: (B, C, H, W) -> (B, C, H*W)
        batch_size = logits.size(0)
        num_classes = logits.size(1)

        probs = probs.view(batch_size, num_classes, -1)
        targets = targets.view(batch_size, num_classes, -1)

        # Calculate intersection and union
        intersection = (probs * targets).sum(dim=2)
        total = probs.sum(dim=2) + targets.sum(dim=2)

        # Calculate Dice score
        dice_score = (2.0 * intersection + self.smooth) / (total + self.smooth)

        # Return 1 - Dice (Loss)
        # Average over classes and batch
        return 1 - dice_score.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (float): Weighting factor for the rare class (foreground).
        gamma (float): Focusing parameter to down-weight easy examples.
        reduction (str): 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model output of shape (B, C, H, W).
            targets (torch.Tensor): Ground truth of shape (B, C, H, W).
        """
        # BCEWithLogitsLoss is numerically stable
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # p = sigmoid(logits)
        # if y=1, p_t = p; if y=0, p_t = 1-p
        # This is equivalent to exp(-bce_loss)
        p_t = torch.exp(-bce_loss)

        # Calculate Focal Loss term
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ComboLoss(nn.Module):
    """
    Combination of Dice Loss and Focal Loss.
    Loss = dice_weight * DiceLoss + focal_weight * FocalLoss
    """

    def __init__(
        self, dice_weight=0.5, focal_weight=0.5, smooth=1.0, alpha=0.25, gamma=2.0
    ):
        super(ComboLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return self.dice_weight * dice + self.focal_weight * focal
