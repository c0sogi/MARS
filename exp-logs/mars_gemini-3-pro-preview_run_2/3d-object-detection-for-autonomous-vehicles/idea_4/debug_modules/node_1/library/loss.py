import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def _transpose_and_gather_feat(feat, ind):
    """
    Transposes the feature map and gathers values at specific indices.
    Args:
        feat: (B, C, H, W) feature map
        ind: (B, K) indices of ground truth centers
    Returns:
        feat: (B, K, C) gathered features
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
    feat = feat.view(feat.size(0), -1, feat.size(3))  # (B, H*W, C)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))  # (B, K, C)
    feat = feat.gather(1, ind)  # (B, K, C)
    return feat


def _neg_loss(pred, gt):
    """
    Modified Focal Loss for Heatmap Regression (Penalty-reduced pixel-wise logistic regression).
    Args:
        pred: (B, C, H, W) predicted heatmap (sigmoid activated)
        gt: (B, C, H, W) ground truth heatmap
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


def _reg_loss(regr, gt_regr, mask):
    """
    L1 Regression Loss masked by object existence.
    Args:
        regr: (B, K, C) predicted regression values
        gt_regr: (B, K, C) ground truth regression values
        mask: (B, K) boolean mask indicating presence of objects
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = F.l1_loss(regr, gt_regr, reduction="sum")
    loss = loss / (num + 1e-4)
    return loss


class Mono3DLoss(nn.Module):
    def __init__(self):
        super(Mono3DLoss, self).__init__()
        # Loss weights (standard CenterNet configuration)
        self.hm_weight = 1.0
        self.dim_weight = 0.1
        self.depth_weight = 0.1
        self.rot_weight = 1.0
        self.off_weight = 1.0

    def forward(self, outputs, batch):
        """
        Compute the total loss.
        Args:
            outputs: Dictionary containing model outputs:
                     'hm': (B, C, H, W)
                     'dim': (B, 3, H, W)
                     'depth': (B, 1, H, W)
                     'rot': (B, 2, H, W)
                     'offset': (B, 2, H, W)
            batch: Dictionary containing ground truth targets:
                   'hm': (B, C, H, W)
                   'ind': (B, K)
                   'reg_mask': (B, K)
                   'dim': (B, K, 3)
                   'depth': (B, K, 1)
                   'rot': (B, K, 2)
                   'offset': (B, K, 2)
        """

        # 1. Heatmap Loss
        hm_pred = torch.sigmoid(outputs["hm"])
        hm_loss = _neg_loss(hm_pred, batch["hm"])

        # 2. Gather predictions at ground truth centers
        # We need to extract the values from the specific pixels where objects are located
        ind = batch["ind"]

        dim_pred = _transpose_and_gather_feat(outputs["dim"], ind)
        depth_pred = _transpose_and_gather_feat(outputs["depth"], ind)
        rot_pred = _transpose_and_gather_feat(outputs["rot"], ind)
        off_pred = _transpose_and_gather_feat(outputs["offset"], ind)

        # 3. Regression Losses
        mask = batch["reg_mask"]

        dim_loss = _reg_loss(dim_pred, batch["dim"], mask)
        depth_loss = _reg_loss(depth_pred, batch["depth"], mask)
        rot_loss = _reg_loss(rot_pred, batch["rot"], mask)
        off_loss = _reg_loss(off_pred, batch["offset"], mask)

        # 4. Weighted Sum
        loss = (
            self.hm_weight * hm_loss
            + self.dim_weight * dim_loss
            + self.depth_weight * depth_loss
            + self.rot_weight * rot_loss
            + self.off_weight * off_loss
        )

        # Return loss and stats for logging
        loss_stats = {
            "loss": loss,
            "hm_loss": hm_loss,
            "dim_loss": dim_loss,
            "depth_loss": depth_loss,
            "rot_loss": rot_loss,
            "off_loss": off_loss,
        }

        return loss, loss_stats
