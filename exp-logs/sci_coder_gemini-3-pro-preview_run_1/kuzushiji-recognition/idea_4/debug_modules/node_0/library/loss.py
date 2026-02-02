import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterNetLoss(nn.Module):
    def __init__(self):
        super(CenterNetLoss, self).__init__()

    def modified_focal_loss(self, pred, gt):
        """
        Modified focal loss for heatmap.
        Args:
            pred (torch.Tensor): (batch, 1, h, w) - raw logits (will be sigmoid-ed)
            gt (torch.Tensor): (batch, 1, h, w) - gaussian heatmap (0-1)
        """
        pred = torch.sigmoid(pred)

        # Identify positive and negative samples
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # Weighting for negative samples (closer to center -> less penalty)
        neg_weights = torch.pow(1 - gt, 4)

        # Clamp for numerical stability
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        # Loss calculation
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos

    def reg_l1_loss(self, pred, target, mask):
        """
        L1 regression loss.
        Args:
            pred (torch.Tensor): (batch, max_objs, 2)
            target (torch.Tensor): (batch, max_objs, 2)
            mask (torch.Tensor): (batch, max_objs)
        """
        expand_mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss

    def cls_loss(self, pred, target, mask):
        """
        Classification loss (Cross Entropy).
        Args:
            pred (torch.Tensor): (batch, max_objs, num_classes)
            target (torch.Tensor): (batch, max_objs) - class indices
            mask (torch.Tensor): (batch, max_objs)
        """
        # Flatten batch and objects for cross entropy
        pred = pred.view(-1, pred.size(2))  # (B*N, C)
        target = target.view(-1)  # (B*N)
        mask = mask.view(-1).float()  # (B*N)

        # Calculate CE loss per element, then mask
        loss = F.cross_entropy(pred, target, reduction="none")
        loss = (loss * mask).sum() / (mask.sum() + 1e-4)
        return loss

    def _gather_feat(self, feat, ind):
        """
        Gather features at specific indices.
        Args:
            feat (torch.Tensor): (B, C, H, W)
            ind (torch.Tensor): (B, K)
        """
        dim = feat.size(1)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

        # Flatten spatial dims: (B, C, H*W) -> (B, H*W, C)
        feat = feat.view(feat.size(0), dim, -1).permute(0, 2, 1)

        # Gather
        feat = feat.gather(1, ind)
        return feat

    def forward(self, outputs, batch):
        """
        Calculate total loss.
        Args:
            outputs (dict): Model outputs {'hm', 'cls', 'wh', 'reg'}
            batch (dict): Ground truth batch
        """
        hm_pred = outputs["hm"]
        cls_pred = outputs["cls"]
        wh_pred = outputs["wh"]
        reg_pred = outputs["reg"]

        # Move targets to same device as predictions
        hm_gt = batch["hm"].to(hm_pred.device)
        wh_gt = batch["wh"].to(wh_pred.device)
        reg_gt = batch["reg"].to(reg_pred.device)
        ind = batch["ind"].to(reg_pred.device)
        cat_gt = batch["cat"].to(reg_pred.device)
        reg_mask = batch["reg_mask"].to(reg_pred.device)

        # 1. Heatmap Loss
        loss_hm = self.modified_focal_loss(hm_pred, hm_gt)

        # 2. Gather predictions at ground truth centers
        wh_pred_gathered = self._gather_feat(wh_pred, ind)
        reg_pred_gathered = self._gather_feat(reg_pred, ind)
        cls_pred_gathered = self._gather_feat(cls_pred, ind)

        # 3. Regression Losses
        loss_wh = self.reg_l1_loss(wh_pred_gathered, wh_gt, reg_mask)
        loss_reg = self.reg_l1_loss(reg_pred_gathered, reg_gt, reg_mask)

        # 4. Classification Loss
        loss_cls = self.cls_loss(cls_pred_gathered, cat_gt, reg_mask)

        # Weighted sum (Standard CenterNet weights)
        # hm=1, wh=0.1, reg=1, cls=1
        loss = loss_hm + 0.1 * loss_wh + 1.0 * loss_reg + 1.0 * loss_cls

        loss_stats = {
            "loss": loss,
            "hm_loss": loss_hm,
            "wh_loss": loss_wh,
            "reg_loss": loss_reg,
            "cls_loss": loss_cls,
        }

        return loss, loss_stats
