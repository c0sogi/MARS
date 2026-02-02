import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def _transpose_and_gather_feat(feat, ind):
    """
    Extracts features from specific indices in the dense feature map.
    Args:
        feat: (B, C, H, W) Dense feature map
        ind: (B, K) Indices of the objects (flattened H*W index)
    Returns:
        feat: (B, K, C) Gathered features
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
    feat = feat.view(feat.size(0), -1, feat.size(3))  # (B, H*W, C)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))  # (B, K, C)
    feat = feat.gather(1, ind)  # (B, K, C)
    return feat


def _neg_loss(pred, gt):
    """
    Modified Focal Loss for Heatmap.
    Args:
        pred: (B, C, H, W) Logits from the model
        gt: (B, C, H, W) Ground truth heatmap (0-1)
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    # Apply sigmoid to convert logits to probabilities
    pred = torch.sigmoid(pred)

    # Clamp for numerical stability
    pred = torch.clamp(pred, 1e-12, 1 - 1e-12)

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
        regr: (B, K, C) Predicted regression values
        gt_regr: (B, K, C) Ground truth regression values
        mask: (B, K) Mask indicating valid objects (1=object, 0=empty)
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = F.l1_loss(regr, gt_regr, reduction="sum")
    loss = loss / (num + 1e-4)
    return loss


class CenterNetLoss(nn.Module):
    def __init__(self):
        super(CenterNetLoss, self).__init__()
        self.loss_weights = Config.LOSS_WEIGHTS

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dictionary of model outputs (logits).
                     keys: 'hm', 'reg', 'wh', 'depth', 'rot'
            targets: Dictionary of ground truth targets.
                     keys: 'hm', 'ind', 'mask', 'reg', 'wh', 'depth', 'rot'
        """
        loss_stats = {}
        total_loss = 0.0

        # 1. Heatmap Loss
        if "hm" in outputs:
            hm_loss = _neg_loss(outputs["hm"], targets["hm"])
            loss_stats["hm_loss"] = hm_loss.item()
            total_loss += self.loss_weights["hm"] * hm_loss

        # 2. Regression Losses
        # We need to gather the predictions corresponding to the ground truth indices

        # Helper list for regression heads
        reg_heads = ["reg", "wh", "depth", "rot"]

        for head in reg_heads:
            if head in outputs and head in targets:
                # Extract prediction at object centers
                pred = _transpose_and_gather_feat(outputs[head], targets["ind"])

                # Calculate masked L1 loss
                l1_loss = _reg_loss(pred, targets[head], targets["mask"])

                loss_stats[f"{head}_loss"] = l1_loss.item()
                total_loss += self.loss_weights[head] * l1_loss

        loss_stats["total_loss"] = total_loss.item()

        return total_loss, loss_stats
