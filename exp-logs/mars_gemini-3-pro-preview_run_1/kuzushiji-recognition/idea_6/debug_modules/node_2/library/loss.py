import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_feat(feat, ind, mask=None):
    """
    Gather features from a flattened feature map at specific indices.

    Args:
        feat (torch.Tensor): The feature map of shape (B, H*W, C).
        ind (torch.Tensor): The indices of shape (B, K).
        mask (torch.Tensor, optional): Mask of shape (B, K).

    Returns:
        torch.Tensor: Gathered features of shape (B, K, C) or flattened if mask is used.
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
    Transposes a feature map and gathers specific indices.
    Optimized to avoid OOM by gathering before permuting channels.

    Args:
        feat (torch.Tensor): Feature map of shape (B, C, H, W).
        ind (torch.Tensor): Indices of shape (B, K).

    Returns:
        torch.Tensor: Gathered features of shape (B, K, C).
    """
    # feat: (B, C, H, W) -> (B, C, N)
    B, C, H, W = feat.shape
    feat = feat.view(B, C, -1)

    # ind: (B, K) -> (B, C, K)
    # We need to expand indices across the channel dimension
    ind_expanded = ind.unsqueeze(1).expand(B, C, ind.size(1))

    # Gather: (B, C, K)
    feat = feat.gather(2, ind_expanded)

    # Permute to (B, K, C) and ensure contiguous memory for downstream .view() ops
    feat = feat.permute(0, 2, 1).contiguous()

    return feat


class ModifiedFocalLoss(nn.Module):
    """
    Penalty-reduced pixel-wise logistic regression with focal loss.
    Used for the heatmap branch in CenterNet.
    """

    def __init__(self, alpha=2, beta=4):
        super(ModifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Sigmoid output of shape (B, 1, H, W).
            gt (torch.Tensor): Ground truth heatmap of shape (B, 1, H, W).
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.beta)

        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = (
            torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        )

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
    L1 loss for regression tasks (Width/Height and Offsets), masked by object existence.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred (torch.Tensor): Predictions of shape (B, K, 2).
            target (torch.Tensor): Targets of shape (B, K, 2).
            mask (torch.Tensor): Mask of shape (B, K) indicating valid objects.
        """
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        # Normalize by number of objects (plus epsilon to avoid div by zero)
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CenterNetLoss(nn.Module):
    """
    Composite loss function for CenterNet.
    Combines Heatmap Focal Loss, Regression L1 Loss, and Sparse Cross Entropy Loss.
    """

    def __init__(self, hm_weight=1.0, wh_weight=0.1, off_weight=1.0, cls_weight=1.0):
        super(CenterNetLoss, self).__init__()
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight
        self.cls_weight = cls_weight

        self.crit_hm = ModifiedFocalLoss()
        self.crit_reg = RegL1Loss()
        # CrossEntropy is computed functionally

    def forward(self, outputs, batch):
        """
        Args:
            outputs (dict): Model outputs containing 'hm', 'wh', 'reg', 'cls_logits'.
            batch (dict): Batch dictionary containing 'target'.

        Returns:
            tuple: (total_loss, loss_stats_dict)
        """
        targets = batch["target"]

        # 1. Heatmap Loss (Global)
        hm_loss = self.crit_hm(outputs["hm"], targets["hm"])

        # Prepare gathered features for regression and classification
        ind = targets["ind"]

        # 2. Width/Height Regression Loss (Sparse)
        pred_wh = _transpose_and_gather_feat(outputs["wh"], ind)
        wh_loss = self.crit_reg(pred_wh, targets["wh"], targets["reg_mask"])

        # 3. Offset Regression Loss (Sparse)
        pred_reg = _transpose_and_gather_feat(outputs["reg"], ind)
        off_loss = self.crit_reg(pred_reg, targets["reg"], targets["reg_mask"])

        # 4. Classification Loss (Sparse)
        # Gather logits: (B, K, C)
        pred_cls = _transpose_and_gather_feat(outputs["cls_logits"], ind)

        # Flatten tensors to apply masking efficiently
        B, K, C = pred_cls.shape
        pred_cls_flat = pred_cls.view(-1, C)
        target_cls_flat = targets["cls_ids"].view(-1)
        mask_flat = targets["reg_mask"].view(-1) > 0

        # Only compute CrossEntropy on valid objects
        if mask_flat.sum() > 0:
            valid_logits = pred_cls_flat[mask_flat]
            valid_targets = target_cls_flat[mask_flat]
            cls_loss = F.cross_entropy(valid_logits, valid_targets)
        else:
            cls_loss = torch.tensor(0.0, device=pred_cls.device)

        # Total Loss
        loss = (
            self.hm_weight * hm_loss
            + self.wh_weight * wh_loss
            + self.off_weight * off_loss
            + self.cls_weight * cls_loss
        )

        loss_stats = {
            "loss": loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
            "cls_loss": cls_loss,
        }

        return loss, loss_stats
