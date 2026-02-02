import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def _transpose_and_gather_feat(feat, ind):
    """
    Extracts features from specific spatial locations.
    feat: (Batch, Channels, Height, Width)
    ind: (Batch, K) - Indices of the objects in the flattened spatial dimension
    Returns: (Batch, K, Channels)
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = torch.gather(
        feat, 1, ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))
    )
    return feat


class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss for Heatmap regression (CornerNet/CenterNet variant).
    Penalizes deviations from the Gaussian ground truth.
    """

    def __init__(self):
        super(ModifiedFocalLoss, self).__init__()

    def forward(self, pred, gt):
        """
        pred: (Batch, Classes, H, W) - Sigmoid output
        gt: (Batch, Classes, H, W) - Gaussian rendered targets [0, 1]
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        # Clamp predictions to avoid log(0)
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
    L1 Loss for regression heads (Size and Offset), applied only at object centers.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        """
        pred: (Batch, K, 2) - Gathered predictions
        target: (Batch, K, 2) - Gathered targets
        mask: (Batch, K) - Binary mask indicating valid objects
        """
        expand_mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction="sum")

        # Normalize by number of objects (plus epsilon)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CenterNetLoss(nn.Module):
    """
    Composite loss function for the Anatomically-Aware CenterNet.
    Combines:
    1. Heatmap Focal Loss
    2. Size (WH) L1 Loss
    3. Offset (Reg) L1 Loss
    4. Global Classification BCE Loss
    """

    def __init__(self):
        super(CenterNetLoss, self).__init__()
        self.crit = ModifiedFocalLoss()
        self.crit_reg = RegL1Loss()

        # Weights
        self.lambda_hm = Config.LAMBDA_HEATMAP
        self.lambda_wh = Config.LAMBDA_SIZE
        self.lambda_off = Config.LAMBDA_OFFSET
        self.lambda_global = Config.LAMBDA_GLOBAL

    def forward(self, outputs, batch):
        """
        outputs: Dictionary containing model predictions
            - 'hm': (B, C, H, W)
            - 'wh': (B, 2, H, W)
            - 'reg': (B, 2, H, W)
            - 'global_label': (B, 1)
        batch: Dictionary containing targets
            - 'hm': (B, C, H, W)
            - 'wh': (B, 2, H, W)
            - 'reg': (B, 2, H, W)
            - 'ind': (B, K)
            - 'reg_mask': (B, K)
            - 'global_label': (B, 1)
        """

        # 1. Heatmap Loss
        hm_loss = self.crit(outputs["hm"], batch["hm"])

        # 2. Regression Losses (Size and Offset)
        # We need to gather the values from the dense maps at the specific object indices

        # Gather predictions
        wh_pred = _transpose_and_gather_feat(outputs["wh"], batch["ind"])
        reg_pred = _transpose_and_gather_feat(outputs["reg"], batch["ind"])

        # Gather targets (The dataset provides dense maps, so we gather from them too for consistency,
        # though we could have passed the sparse values directly if the dataset was designed that way.
        # Based on dataset.py, 'wh' and 'reg' are dense maps).
        wh_target = _transpose_and_gather_feat(batch["wh"], batch["ind"])
        reg_target = _transpose_and_gather_feat(batch["reg"], batch["ind"])

        mask = batch["reg_mask"]

        wh_loss = self.crit_reg(wh_pred, wh_target, mask)
        off_loss = self.crit_reg(reg_pred, reg_target, mask)

        # 3. Global Classification Loss
        # outputs['global_label'] are logits
        # batch['global_label'] is 0.0 or 1.0
        global_pred = outputs["global_label"]
        global_target = batch["global_label"]

        # Use BCEWithLogitsLoss for stability
        global_loss = F.binary_cross_entropy_with_logits(global_pred, global_target)

        # 4. Total Loss
        loss = (
            self.lambda_hm * hm_loss
            + self.lambda_wh * wh_loss
            + self.lambda_off * off_loss
            + self.lambda_global * global_loss
        )

        loss_stats = {
            "loss": loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
            "global_loss": global_loss,
        }

        return loss, loss_stats
