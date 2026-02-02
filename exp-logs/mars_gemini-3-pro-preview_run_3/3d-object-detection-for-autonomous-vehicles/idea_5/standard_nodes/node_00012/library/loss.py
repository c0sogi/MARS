import torch
import torch.nn as nn
import torch.nn.functional as F
from library.utils import _transpose_and_gather_feat


class FastFocalLoss(nn.Module):
    """
    Modified Focal Loss for CenterNet heatmap regression.
    Penalizes easy negative examples and reduces penalty around ground truth centers
    using a Gaussian kernel (encoded in the target).
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super(FastFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        Args:
            pred (Tensor): Prediction logits of shape (B, C, H, W).
            gt (Tensor): Ground truth heatmap of shape (B, C, H, W) with values in [0, 1].

        Returns:
            Tensor: Scalar loss value.
        """
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)

        # Clamp for numerical stability
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        # Identify positive (center) and negative (background/nearby) indices
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # Weighting for negative examples based on distance to GT center (beta)
        neg_weights = torch.pow(1 - gt, self.beta)

        # Focal loss calculation
        # Loss for positive examples: - (1 - pred)^alpha * log(pred)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds

        # Loss for negative examples: - (1 - gt)^beta * pred^alpha * log(1 - pred)
        neg_loss = (
            torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        )

        # Normalize by number of positive samples
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss


class RegLoss(nn.Module):
    """
    Masked L1 Loss for regression heads (dimensions, offsets, rotation, z).
    Only calculates loss at the specific indices of ground truth objects.
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, pred, target, mask, ind):
        """
        Args:
            pred (Tensor): Dense prediction map (B, C, H, W).
            target (Tensor): Dense target map (B, C, H, W).
            mask (Tensor): Mask of valid objects (B, K). 1 for object, 0 for padding.
            ind (Tensor): Indices of object centers in flattened map (B, K).

        Returns:
            Tensor: Scalar loss value.
        """
        # Gather predictions and targets at the specific object indices
        # Shape changes from (B, C, H, W) -> (B, K, C)
        pred_gathered = _transpose_and_gather_feat(pred, ind)
        target_gathered = _transpose_and_gather_feat(target, ind)

        # Calculate L1 loss per element
        loss = F.l1_loss(pred_gathered, target_gathered, reduction="none")

        # Sum loss across channel dimension (C)
        loss = loss.sum(dim=2)

        # Apply mask to ignore padded objects
        loss = loss * mask

        # Normalize by the number of valid objects
        # Add epsilon to prevent division by zero
        normalizer = mask.sum() + 1e-4
        loss = loss.sum() / normalizer

        return loss
