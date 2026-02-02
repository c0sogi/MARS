import torch
import torch.nn as nn
import torch.nn.functional as F


class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss for CenterNet Heatmap Regression.

    This loss function is a variant of Focal Loss used in CornerNet and CenterNet.
    It penalizes negative samples less if they are close to the ground truth center
    (indicated by the gaussian values in the ground truth heatmap).
    """

    def __init__(self):
        super(ModifiedFocalLoss, self).__init__()

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Predicted heatmap (B, C, H, W), values in [0, 1].
            gt (torch.Tensor): Ground truth heatmap (B, C, H, W), values in [0, 1].
                               Contains gaussian peaks at object centers.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Identify positive (center) and negative (background/gaussian slope) samples
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # Weight for negative samples based on proximity to ground truth
        # Closer to center (higher gt value) -> lower weight -> less penalty
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        # Loss for positive samples: (1 - pred)^alpha * log(pred)
        # alpha = 2
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds

        # Loss for negative samples: (1 - gt)^beta * pred^alpha * log(1 - pred)
        # alpha = 2, beta = 4
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        # Normalize by number of positive samples
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss


class RegL1Loss(nn.Module):
    """
    Masked L1 Loss for Regression Heads (Size and Offset).

    Computes L1 loss only at the indices corresponding to object centers.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred (torch.Tensor): Predicted regression values (B, C, H, W).
            target (torch.Tensor): Ground truth regression values (B, C, H, W).
            mask (torch.Tensor): Binary mask indicating object centers (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Expand mask to match the channel dimension of predictions (e.g., 2 for width/height)
        mask = mask.expand_as(pred)

        # Compute L1 loss only where mask is 1
        # We multiply by mask to zero out background predictions/targets
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")

        # Normalize by the number of items in the mask
        # Add epsilon to prevent division by zero
        mask_sum = mask.sum() + 1e-4

        loss = loss / mask_sum

        return loss
