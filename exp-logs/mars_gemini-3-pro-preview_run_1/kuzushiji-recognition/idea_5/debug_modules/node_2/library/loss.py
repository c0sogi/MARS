import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.utils import _transpose_and_gather_feat


class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss from CornerNet/CenterNet.
    Penalizes the background predictions less around the ground truth centers
    based on a Gaussian distribution.
    """

    def __init__(self, alpha=2, beta=4):
        super(ModifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Predicted heatmap (B, C, H, W), values in [0, 1].
            gt (torch.Tensor): Ground truth heatmap (B, C, H, W), values in [0, 1].
        """
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0

        # Positive loss: - (1 - pred)^alpha * log(pred)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds

        # Negative loss: - (1 - gt)^beta * pred^alpha * log(1 - pred)
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
    L1 Loss for regression tasks (width/height and offset).
    Applied only at ground truth indices.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred (torch.Tensor): Predicted values gathered at indices (B, K, C).
            target (torch.Tensor): Ground truth values (B, K, C).
            mask (torch.Tensor): Mask indicating valid objects (B, K).
        """
        expand_mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction="sum")

        # Normalize by the number of objects
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CenterNetLoss(nn.Module):
    def __init__(self):
        super(CenterNetLoss, self).__init__()
        self.crit_hm = ModifiedFocalLoss()
        self.crit_reg = RegL1Loss()
        # CrossEntropyLoss for classification
        self.crit_cls = nn.CrossEntropyLoss(reduction="none")

        # Weights
        self.hm_weight = Config.HM_WEIGHT
        self.wh_weight = Config.WH_WEIGHT
        self.off_weight = Config.OFF_WEIGHT
        self.cls_weight = 1.0  # Default weight for classification

    def forward(self, outputs, batch):
        """
        Args:
            outputs (dict): Model outputs containing:
                - 'hm': Heatmap (B, 1, H, W)
                - 'wh': Size regression (B, 2, H, W)
                - 'reg': Offset regression (B, 2, H, W)
                - 'cls': Classification logits (B, NumClasses, H, W)
            batch (dict): Batch data containing:
                - 'heatmap': GT heatmap (B, 1, H, W)
                - 'ind': Indices of objects (B, K)
                - 'mask': Mask for valid objects (B, K)
                - 'wh': GT width/height (B, K, 2) - Note: dataset returns dense (B, 2, H, W) but we gather
                - 'reg': GT offset (B, K, 2) - Note: dataset returns dense (B, 2, H, W) but we gather
                - 'cls_ids': GT class IDs (B, K)
        """

        hm_pred = outputs["hm"]
        wh_pred = outputs["wh"]
        reg_pred = outputs["reg"]
        cls_pred = outputs["cls"]

        hm_target = batch["heatmap"].to(Config.DEVICE)
        ind = batch["ind"].to(Config.DEVICE)
        mask = batch["mask"].to(Config.DEVICE)
        cls_target = batch["cls_ids"].to(Config.DEVICE)

        # 1. Heatmap Loss
        # Ensure sigmoid is applied if model outputs logits (usually handled in model,
        # but safe to clamp/sigmoid here if needed. Assuming model output is sigmoid-ed or
        # we treat it as probabilities).
        # Standard CenterNet models output sigmoid.
        hm_loss = self.crit_hm(hm_pred, hm_target)

        # 2. Regression Losses (WH and Offset)
        # Gather predictions at ground truth locations
        wh_pred_gathered = _transpose_and_gather_feat(wh_pred, ind)
        reg_pred_gathered = _transpose_and_gather_feat(reg_pred, ind)

        # Gather targets from dense maps provided by dataset
        # Dataset provides dense maps for wh and reg in 'wh' and 'reg' keys
        # We need to gather them using indices to match the prediction shape (B, K, 2)
        # Alternatively, the dataset might provide them pre-gathered if we look closely at dataset.py?
        # Looking at dataset.py: 'wh' and 'reg' are dense maps (2, H, W).
        # We need to gather the GT values from these maps using the same indices.

        wh_target_dense = batch["wh"].to(Config.DEVICE)
        reg_target_dense = batch["reg"].to(Config.DEVICE)

        wh_target_gathered = _transpose_and_gather_feat(wh_target_dense, ind)
        reg_target_gathered = _transpose_and_gather_feat(reg_target_dense, ind)

        wh_loss = self.crit_reg(wh_pred_gathered, wh_target_gathered, mask)
        off_loss = self.crit_reg(reg_pred_gathered, reg_target_gathered, mask)

        # 3. Classification Loss
        # Gather classification logits at ground truth locations
        # cls_pred: (B, NumClasses, H, W)
        cls_pred_gathered = _transpose_and_gather_feat(
            cls_pred, ind
        )  # (B, K, NumClasses)

        # Flatten for CrossEntropy
        # cls_pred_gathered: (B*K, NumClasses)
        # cls_target: (B, K) -> (B*K)
        cls_pred_flat = cls_pred_gathered.view(-1, Config.NUM_CLASSES)
        cls_target_flat = cls_target.view(-1)
        mask_flat = mask.view(-1)

        # Calculate CE loss
        cls_loss_raw = self.crit_cls(cls_pred_flat, cls_target_flat)

        # Apply mask and normalize
        cls_loss = (cls_loss_raw * mask_flat).sum() / (mask_flat.sum() + 1e-4)

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
