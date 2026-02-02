import torch
import torch.nn as nn
import torch.nn.functional as F


def _transpose_and_gather_feat(feat, ind):
    """
    Transposes the feature map and gathers specific indices.

    Args:
        feat (torch.Tensor): Feature map of shape [B, C, H, W]
        ind (torch.Tensor): Indices of shape [B, K]

    Returns:
        torch.Tensor: Gathered features of shape [B, K, C]
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))
    feat = feat.gather(1, ind)
    return feat


class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss for CenterNet Heatmap.
    Penalizes deviations from the ground truth Gaussian heatmap.
    """

    def __init__(self):
        super(ModifiedFocalLoss, self).__init__()

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Predicted logits [B, C, H, W]
            gt (torch.Tensor): Ground truth heatmap [B, C, H, W] (values 0-1)
        """
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)

        # Clamp for numerical stability
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # Weight for negative samples decreases as they get closer to the peak
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        # Loss for positive peaks
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds

        # Loss for background/surrounding pixels
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

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
    L1 Loss for regression tasks (Size and Offset), applied only at object centers.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask, ind):
        """
        Args:
            pred (torch.Tensor): Prediction map [B, C, H, W]
            target (torch.Tensor): Ground truth values [B, K, C]
            mask (torch.Tensor): Mask indicating valid objects [B, K]
            ind (torch.Tensor): Indices of object centers [B, K]
        """
        # Extract predictions at the specific object locations
        pred = _transpose_and_gather_feat(pred, ind)

        # Apply mask
        mask = mask.unsqueeze(2).expand_as(pred).float()

        # Calculate L1 loss
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")

        # Normalize by number of objects
        loss = loss / (mask.sum() + 1e-4)
        return loss


class MultiTaskLoss(nn.Module):
    """
    Composite loss function for Multi-Task CenterNet.
    Combines Heatmap Focal Loss, Regression L1 Loss, and Global Classification BCE Loss.
    """

    def __init__(self, hm_weight=1.0, wh_weight=0.1, off_weight=1.0, global_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight
        self.global_weight = global_weight

        self.focal_loss = ModifiedFocalLoss()
        self.l1_loss = RegL1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, batch):
        """
        Args:
            outputs (dict): Model outputs containing 'hm', 'wh', 'reg', 'global_logits'
            batch (dict): Batch targets containing 'hm', 'wh', 'reg', 'ind', 'reg_mask', 'global_target'

        Returns:
            tuple: (total_loss, loss_stats_dict)
        """
        # 1. Heatmap Loss (Focal)
        hm_loss = self.focal_loss(outputs["hm"], batch["hm"])

        # 2. Size (Width/Height) Loss (L1)
        wh_loss = self.l1_loss(
            outputs["wh"], batch["wh"], batch["reg_mask"], batch["ind"]
        )

        # 3. Offset Loss (L1)
        off_loss = self.l1_loss(
            outputs["reg"], batch["reg"], batch["reg_mask"], batch["ind"]
        )

        # 4. Global Classification Loss (BCE)
        # Ensure target is float and correct shape [B, 1]
        global_target = batch["global_target"].float().view(-1, 1)
        global_loss = self.bce_loss(outputs["global_logits"], global_target)

        # Weighted Sum
        total_loss = (
            self.hm_weight * hm_loss
            + self.wh_weight * wh_loss
            + self.off_weight * off_loss
            + self.global_weight * global_loss
        )

        loss_stats = {
            "loss": total_loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
            "global_loss": global_loss,
        }

        return total_loss, loss_stats
