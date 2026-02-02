import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_feat(feat, ind, mask=None):
    """
    Gathers values from the feature map at specific indices.

    Args:
        feat (torch.Tensor): The feature map (B, H*W, C) or similar.
        ind (torch.Tensor): The indices to gather (B, K).
        mask (torch.Tensor, optional): Mask to filter gathered values.

    Returns:
        torch.Tensor: Gathered features.
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
    Transposes the feature map and gathers values at specific indices.

    Args:
        feat (torch.Tensor): Feature map of shape (B, C, H, W).
        ind (torch.Tensor): Indices of shape (B, K).

    Returns:
        torch.Tensor: Gathered features of shape (B, K, C).
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class CenterNetLoss(nn.Module):
    def __init__(self, wh_weight=0.1, reg_weight=1.0):
        """
        Initializes the CenterNet loss module.

        Args:
            wh_weight (float): Weight for the width/height regression loss.
            reg_weight (float): Weight for the offset regression loss.
        """
        super(CenterNetLoss, self).__init__()
        self.wh_weight = wh_weight
        self.reg_weight = reg_weight

    def focal_loss(self, preds, targets):
        """
        Modified focal loss for heatmap prediction.

        Args:
            preds (torch.Tensor): Predicted heatmap logits (B, C, H, W).
            targets (torch.Tensor): Ground truth heatmap (B, C, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        preds = torch.sigmoid(preds)

        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()

        # Penalty reduction for negative samples near the center (Gaussian)
        neg_weights = torch.pow(1 - targets, 4)

        loss = 0

        # Clamp predictions to avoid log(0)
        preds = torch.clamp(preds, 1e-12, 1 - 1e-12)

        # Standard Focal Loss formula components
        # alpha = 2, beta = 4
        pos_loss = torch.log(preds) * torch.pow(1 - preds, 2) * pos_inds
        neg_loss = torch.log(1 - preds) * torch.pow(preds, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss

    def reg_l1_loss(self, preds, targets, mask):
        """
        L1 regression loss applied only at object locations.

        Args:
            preds (torch.Tensor): Predicted values (B, K, 2).
            targets (torch.Tensor): Ground truth values (B, K, 2).
            mask (torch.Tensor): Mask indicating valid objects (B, K).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        expand_mask = mask.unsqueeze(2).expand_as(preds).float()

        loss = F.l1_loss(preds * expand_mask, targets * expand_mask, reduction="sum")

        # Normalize by the number of objects
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def forward(self, outputs, targets):
        """
        Calculates the total loss.

        Args:
            outputs (dict): Dictionary containing model outputs:
                - 'hm': Heatmap logits (B, C, H, W)
                - 'wh': Size predictions (B, 2, H, W)
                - 'reg': Offset predictions (B, 2, H, W)
            targets (dict): Dictionary containing ground truth:
                - 'hm': Heatmap targets
                - 'wh': Size targets
                - 'reg': Offset targets
                - 'ind': Indices of objects
                - 'reg_mask': Mask of valid objects

        Returns:
            tuple: (total_loss, loss_stats_dict)
        """
        hm_pred = outputs["hm"]
        wh_pred = outputs["wh"]
        reg_pred = outputs["reg"]

        hm_target = targets["hm"].to(hm_pred.device)
        wh_target = targets["wh"].to(wh_pred.device)
        reg_target = targets["reg"].to(reg_pred.device)
        ind_target = targets["ind"].to(reg_pred.device)
        mask_target = targets["reg_mask"].to(reg_pred.device)

        # 1. Heatmap Loss
        hm_loss = self.focal_loss(hm_pred, hm_target)

        # 2. Gather predictions at ground truth locations
        # wh_pred: (B, 2, H, W) -> (B, K, 2)
        wh_pred_gathered = _transpose_and_gather_feat(wh_pred, ind_target)
        # reg_pred: (B, 2, H, W) -> (B, K, 2)
        reg_pred_gathered = _transpose_and_gather_feat(reg_pred, ind_target)

        # 3. Regression Losses
        wh_loss = self.reg_l1_loss(wh_pred_gathered, wh_target, mask_target)
        reg_loss = self.reg_l1_loss(reg_pred_gathered, reg_target, mask_target)

        # 4. Total Loss
        total_loss = hm_loss + (self.wh_weight * wh_loss) + (self.reg_weight * reg_loss)

        loss_stats = {
            "loss": total_loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "reg_loss": reg_loss,
        }

        return total_loss, loss_stats
