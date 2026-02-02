import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_feat(feat, ind, mask=None):
    """
    Gather feature at specified index.
    Args:
        feat: (B, N, C)
        ind: (B, K)
        mask: (B, K)
    Returns:
        (B, K, C)
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """
    Transpose feature map and gather features at specific indices.
    Args:
        feat: (B, C, H, W)
        ind: (B, K)
    Returns:
        (B, K, C)
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class FastFocalLoss(nn.Module):
    """
    Penalty-reduced Focal Loss for heatmap regression.
    """

    def __init__(self):
        super(FastFocalLoss, self).__init__()

    def forward(self, pred, gt):
        """
        Args:
            pred: (B, C, H, W) - Sigmoid output
            gt: (B, C, H, W) - Ground truth heatmap
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        # Clamp for numerical stability
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
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
    L1 Loss for regression targets, calculated only at ground truth centers.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, ind, mask):
        """
        Args:
            pred: (B, C, H, W)
            target: (B, K, C_reg)
            ind: (B, K)
            mask: (B, K)
        """
        pred = _transpose_and_gather_feat(pred, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()

        # Sum reduction then normalize by number of objects
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CenterPointLoss(nn.Module):
    """
    Composite loss for CenterPoint 3D Object Detection.
    Combines Heatmap Focal Loss and Regression L1 Loss.
    """

    def __init__(self, hm_weight=1.0, reg_weight=1.0):
        super(CenterPointLoss, self).__init__()
        self.hm_weight = hm_weight
        self.reg_weight = reg_weight
        self.focal_loss = FastFocalLoss()
        self.reg_l1_loss = RegL1Loss()

    def forward(self, preds, targets):
        """
        Args:
            preds: Dict containing 'hm' and 'reg' tensors.
            targets: Dict containing 'hm', 'target_reg', 'ind', 'mask'.
        Returns:
            loss: Scalar tensor
            stats: Dict of loss components
        """
        pred_hm = preds["hm"]
        pred_reg = preds["reg"]

        # Ensure targets are on the correct device
        gt_hm = targets["hm"].to(pred_hm.device)
        gt_reg = targets["target_reg"].to(pred_reg.device)
        gt_ind = targets["ind"].to(pred_reg.device)
        gt_mask = targets["mask"].to(pred_reg.device)

        # 1. Heatmap Loss
        # Apply sigmoid to logits before calculating focal loss
        pred_hm_sigmoid = torch.sigmoid(pred_hm)
        hm_loss = self.focal_loss(pred_hm_sigmoid, gt_hm)

        # 2. Regression Loss
        reg_loss = self.reg_l1_loss(pred_reg, gt_reg, gt_ind, gt_mask)

        # Total Loss
        loss = (self.hm_weight * hm_loss) + (self.reg_weight * reg_loss)

        stats = {
            "hm_loss": hm_loss.item(),
            "reg_loss": reg_loss.item(),
            "total_loss": loss.item(),
        }

        return loss, stats
