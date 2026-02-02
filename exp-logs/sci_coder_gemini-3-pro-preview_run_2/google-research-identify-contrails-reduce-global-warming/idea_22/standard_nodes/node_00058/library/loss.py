import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification/segmentation.
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    This implementation accepts logits as input for numerical stability.
    """

    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        """
        Args:
            gamma (float): Focusing parameter. Higher values down-weight easy examples.
            alpha (float, optional): Balancing factor. If None, no alpha weighting is applied.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw predictions (before sigmoid), shape (N, C, H, W).
            targets (torch.Tensor): Ground truth binary masks, shape (N, C, H, W).
        """
        # Compute binary cross entropy
        # reduction='none' allows us to apply the modulating factor element-wise
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # p_t is the probability of the true class.
        # Since BCE = -log(p_t), we can compute p_t = exp(-BCE)
        p_t = torch.exp(-bce_loss)

        # Modulating factor: (1 - p_t)^gamma
        focal_term = (1.0 - p_t) ** self.gamma

        loss = focal_term * bce_loss

        # Apply alpha weighting if specified
        if self.alpha is not None:
            # alpha_t = alpha if target=1 else (1-alpha)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BatchDiceLoss(nn.Module):
    """
    Batch-Level Dice Loss.
    Computes the Dice coefficient over the entire flattened batch (treating the batch as a single volume).
    This stabilizes gradients for sparse targets compared to sample-averaged Dice.
    """

    def __init__(self, smooth=1e-6):
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Raw predictions (before sigmoid).
            targets (torch.Tensor): Ground truth binary masks.
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten the entire batch
        # view(-1) flattens (N, C, H, W) into a single 1D vector
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Compute Intersection and Union over the whole batch
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Dice Coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Dice Loss
        return 1.0 - dice


class FocalBatchDiceLoss(nn.Module):
    """
    Composite Loss: Focal Loss + Batch-Level Dice Loss.
    Addresses class imbalance (Dice) and hard example mining (Focal).
    """

    def __init__(self, gamma=Config.FOCAL_GAMMA, smooth=1e-6):
        super(FocalBatchDiceLoss, self).__init__()
        self.focal = FocalLoss(gamma=gamma, reduction="mean")
        self.batch_dice = BatchDiceLoss(smooth=smooth)

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.batch_dice(logits, targets)
        return focal_loss + dice_loss
