import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation.

    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Addresses the extreme class imbalance (approx 1:168) by down-weighting
    easy background examples and focusing training on hard negatives (cirrus clouds)
    and sparse positives (contrails).
    """

    def __init__(self, gamma=Config.FOCAL_GAMMA, alpha=0.25, reduction="mean"):
        """
        Args:
            gamma (float): Focusing parameter. Higher values focus more on hard examples.
            alpha (float): Balancing parameter for the positive class.
                           Commonly 0.25 for detection/segmentation tasks.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model (B, C, H, W) or (B, H, W).
            targets (torch.Tensor): Binary ground truth masks (same shape as inputs).

        Returns:
            torch.Tensor: Computed Focal Loss.
        """
        # Ensure inputs and targets are float
        inputs = inputs.float()
        targets = targets.float()

        # Compute binary cross entropy loss (with logits) per pixel
        # reduction='none' is essential to apply the focal weights element-wise
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Calculate p_t (probability of the true class)
        # p_t = p if y=1 else 1-p
        # Conveniently, p_t = exp(-bce_loss)
        pt = torch.exp(-bce_loss)

        # Calculate Focal term: (1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha balancing
        if self.alpha is not None:
            # alpha_t = alpha if y=1 else (1-alpha)
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


class BatchDiceLoss(nn.Module):
    """
    Batch-Level Dice Loss.

    Formula: 1 - (2 * |X n Y| + smooth) / (|X| + |Y| + smooth)

    Unlike standard Dice Loss which averages scores per image, this computes
    the Dice score over the entire flattened batch (treating the batch as a single volume).
    This stabilizes gradients, especially when many images in a batch might have
    no contrails (empty masks), and aligns strictly with the Global Dice metric.
    """

    def __init__(self, smooth=1.0):
        super(BatchDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model.
            targets (torch.Tensor): Binary ground truth masks.

        Returns:
            torch.Tensor: 1 - Batch Dice Score.
        """
        # Apply sigmoid to convert logits to probabilities
        probs = torch.sigmoid(inputs)

        # Flatten the entire batch to 1D vectors
        # Shape becomes (B * H * W * C,)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # Compute intersection and union over the whole batch
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()

        # Compute Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice


class HybridLoss(nn.Module):
    """
    Combined Loss Function: Focal Loss + Batch Dice Loss.

    Combines the pixel-level hardness mining of Focal Loss with the
    global structural alignment of Batch Dice Loss.
    """

    def __init__(self):
        super(HybridLoss, self).__init__()
        self.focal = FocalLoss(gamma=Config.FOCAL_GAMMA)
        self.batch_dice = BatchDiceLoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model.
            targets (torch.Tensor): Binary ground truth masks.

        Returns:
            torch.Tensor: Sum of Focal Loss and Batch Dice Loss.
        """
        loss_focal = self.focal(inputs, targets)
        loss_dice = self.batch_dice(inputs, targets)

        return loss_focal + loss_dice
