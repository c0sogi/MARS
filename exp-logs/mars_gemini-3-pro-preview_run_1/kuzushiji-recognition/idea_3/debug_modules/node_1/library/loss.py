import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import _transpose_and_gather_feat


class CenterNetLoss(nn.Module):
    """
    Implements the multi-task loss for CenterNet:
    1. Modified Focal Loss for Heatmap (Objectness)
    2. L1 Loss for Size (Width/Height) and Local Offsets
    3. Cross Entropy Loss for Classification at centers
    """

    def __init__(self):
        super(CenterNetLoss, self).__init__()
        self.hm_weight = Config.HM_LOSS_WEIGHT
        self.wh_weight = Config.WH_LOSS_WEIGHT
        self.off_weight = Config.OFF_LOSS_WEIGHT
        self.cls_weight = Config.CLS_LOSS_WEIGHT

    def _neg_loss(self, pred, gt):
        """
        Modified Focal Loss for Heatmap.
        Penalizes background predictions less if they are close to the ground truth.
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # Weight for negative samples based on proximity to GT (gaussian heatmap)
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        # Clamp pred to avoid log(0)
        pred = torch.clamp(pred, min=1e-6, max=1 - 1e-6)

        # Standard focal loss terms
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

    def _reg_loss(self, regr, gt_regr, mask):
        """
        L1 Loss for regression masked by object presence.
        Only computes loss where mask == 1 (at ground truth objects).
        """
        num = mask.float().sum()
        mask = mask.unsqueeze(2).expand_as(gt_regr).float()

        regr = regr * mask
        gt_regr = gt_regr * mask

        regr_loss = F.l1_loss(regr, gt_regr, reduction="sum")
        regr_loss = regr_loss / (num + 1e-4)
        return regr_loss

    def _cls_loss(self, cls_logits, gt_cls_ids, mask):
        """
        Cross Entropy Loss applied only at ground truth centers.
        """
        # cls_logits: (B, K, NumClasses)
        # gt_cls_ids: (B, K)
        # mask: (B, K)

        # Flatten for processing
        cls_logits = cls_logits.view(-1, cls_logits.size(-1))
        gt_cls_ids = gt_cls_ids.view(-1)
        mask = mask.view(-1).float()

        # Compute CE loss (no reduction to apply mask manually)
        loss = F.cross_entropy(cls_logits, gt_cls_ids, reduction="none")

        # Apply mask
        loss = (loss * mask).sum()

        # Normalize by number of positive samples
        num_pos = mask.sum()
        loss = loss / (num_pos + 1e-4)

        return loss

    def forward(self, outputs, batch):
        """
        Computes the total loss.

        Args:
            outputs (dict): Model predictions containing 'hm', 'wh', 'reg', 'cls_logits'.
            batch (dict): Ground truth targets containing 'hm', 'wh', 'reg', 'ind', 'cls_ids', 'reg_mask'.

        Returns:
            total_loss (Tensor): The weighted sum of all losses.
            loss_stats (dict): Dictionary of individual loss components for logging.
        """
        # Predictions
        hm_pred = torch.sigmoid(outputs["hm"])
        wh_pred = outputs["wh"]
        reg_pred = outputs["reg"]
        cls_logits_pred = outputs["cls_logits"]

        # Targets
        # Ensure targets are on the same device as predictions
        device = hm_pred.device
        hm_true = batch["hm"].to(device)
        wh_true = batch["wh"].to(device)
        reg_true = batch["reg"].to(device)
        ind_true = batch["ind"].to(device)
        cls_ids_true = batch["cls_ids"].to(device)
        reg_mask = batch["reg_mask"].to(device)

        # 1. Heatmap Loss
        hm_loss = self._neg_loss(hm_pred, hm_true)

        # 2. Gather features at ground truth centers for regression and classification
        # The model outputs dense maps (H, W), we only care about the values at the object centers.

        # reg_pred: (B, 2, H, W) -> (B, K, 2)
        reg_pred_gathered = _transpose_and_gather_feat(reg_pred, ind_true)

        # wh_pred: (B, 2, H, W) -> (B, K, 2)
        wh_pred_gathered = _transpose_and_gather_feat(wh_pred, ind_true)

        # cls_logits_pred: (B, NumClasses, H, W) -> (B, K, NumClasses)
        cls_pred_gathered = _transpose_and_gather_feat(cls_logits_pred, ind_true)

        # 3. Regression Losses
        off_loss = self._reg_loss(reg_pred_gathered, reg_true, reg_mask)
        wh_loss = self._reg_loss(wh_pred_gathered, wh_true, reg_mask)

        # 4. Classification Loss
        cls_loss = self._cls_loss(cls_pred_gathered, cls_ids_true, reg_mask)

        # Weighted Sum
        total_loss = (
            self.hm_weight * hm_loss
            + self.wh_weight * wh_loss
            + self.off_weight * off_loss
            + self.cls_weight * cls_loss
        )

        return total_loss, {
            "loss": total_loss.item(),
            "hm_loss": hm_loss.item(),
            "wh_loss": wh_loss.item(),
            "off_loss": off_loss.item(),
            "cls_loss": cls_loss.item(),
        }
