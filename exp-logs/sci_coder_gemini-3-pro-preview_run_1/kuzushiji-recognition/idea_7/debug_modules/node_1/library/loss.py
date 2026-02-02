import torch
import torch.nn as nn
import torch.nn.functional as F


class CenterNetLoss(nn.Module):
    def __init__(self, hm_weight=1.0, reg_weight=1.0, cls_weight=1.0):
        super(CenterNetLoss, self).__init__()
        self.hm_weight = hm_weight
        self.reg_weight = reg_weight
        self.cls_weight = cls_weight

        self.l1_loss = nn.L1Loss(reduction="sum")
        self.ce_loss = nn.CrossEntropyLoss(reduction="sum")

    def _gather_feat(self, feat, ind):
        """
        Gathers features from specific indices in the feature map.
        Args:
            feat: (batch, channels, H, W)
            ind: (batch, K) - indices into flattened H*W
        Returns:
            feat: (batch, K, channels)
        """
        b, c, h, w = feat.size()
        # Flatten H, W -> (b, c, h*w) -> (b, h*w, c)
        feat = feat.view(b, c, -1).permute(0, 2, 1).contiguous()
        # Expand indices to gather all channels: (b, K, c)
        ind = ind.unsqueeze(2).expand(b, ind.size(1), c)
        # Gather
        feat = feat.gather(1, ind)
        return feat

    def _modified_focal_loss(self, pred, gt):
        """
        Modified focal loss for heatmaps (from CornerNet/CenterNet).
        Args:
            pred: (batch, 1, h, w) - sigmoid applied
            gt: (batch, 1, h, w) - ground truth gaussian heatmap
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

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

    def forward(self, outputs, batch):
        """
        Args:
            outputs: dict containing 'hm', 'reg_wh', 'cls_logits'
            batch: dict containing 'hm', 'ind', 'reg', 'wh', 'cls_id', 'reg_mask'
        """
        hm_pred = outputs["hm"]  # (B, 1, H, W)
        reg_wh_pred = outputs["reg_wh"]  # (B, 4, H, W)
        cls_pred = outputs["cls_logits"]  # (B, NumClasses, H, W)

        device = hm_pred.device

        hm_true = batch["hm"].to(device)
        ind = batch["ind"].to(device)
        reg_target = batch["reg"].to(device)
        wh_target = batch["wh"].to(device)
        cls_ids = batch["cls_id"].to(device)
        mask = batch["reg_mask"].to(device)

        # --- 1. Heatmap Loss ---
        loss_hm = self._modified_focal_loss(hm_pred, hm_true)

        # --- 2. Regression Loss (Offset + WH) ---
        # Gather predictions at object centers
        # reg_wh_pred is (B, 4, H, W) -> Gather to (B, K, 4)
        pred_reg_wh = self._gather_feat(reg_wh_pred, ind)

        # Concatenate targets: [reg_x, reg_y, w, h]
        target_reg_wh = torch.cat([reg_target, wh_target], dim=2)

        # Mask out invalid objects (padding)
        mask_expanded = mask.unsqueeze(2).expand_as(pred_reg_wh).float()

        loss_reg = self.l1_loss(
            pred_reg_wh * mask_expanded, target_reg_wh * mask_expanded
        )
        loss_reg = loss_reg / (mask_expanded.sum() + 1e-4)

        # --- 3. Classification Loss ---
        # Gather logits: (B, K, NumClasses)
        pred_cls = self._gather_feat(cls_pred, ind)

        # Flatten for CrossEntropy
        # Filter valid objects using mask
        mask_bool = mask.bool()

        valid_pred_cls = pred_cls[mask_bool]  # (N_valid, NumClasses)
        valid_cls_ids = cls_ids[mask_bool]  # (N_valid,)

        if valid_cls_ids.numel() > 0:
            loss_cls = self.ce_loss(valid_pred_cls, valid_cls_ids)
            # Normalize by number of objects
            loss_cls = loss_cls / (valid_cls_ids.numel() + 1e-4)
        else:
            loss_cls = torch.tensor(0.0, device=device)

        # --- Total Loss ---
        total_loss = (
            (self.hm_weight * loss_hm)
            + (self.reg_weight * loss_reg)
            + (self.cls_weight * loss_cls)
        )

        loss_stats = {
            "loss_hm": loss_hm.item(),
            "loss_reg": loss_reg.item(),
            "loss_cls": loss_cls.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, loss_stats
