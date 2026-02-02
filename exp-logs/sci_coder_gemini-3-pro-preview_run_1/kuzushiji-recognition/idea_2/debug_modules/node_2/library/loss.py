import torch
import torch.nn as nn
import torch.nn.functional as F
from library.model import transpose_and_gather_feat


class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss (Penalty Reduced) for CenterNet Heatmap.
    Used to handle class imbalance between center points and background.
    """

    def __init__(self, alpha=2, beta=4):
        super(ModifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Predicted heatmap (Batch, Class, H, W), values in [0, 1].
            gt (torch.Tensor): Ground truth heatmap (Batch, Class, H, W), values in [0, 1].

        Returns:
            torch.Tensor: Scalar loss value.
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
    L1 Loss for regression targets (Width/Height and Offsets).
    Calculated only at ground truth center locations.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred (torch.Tensor): Gathered predictions (Batch, Max_Objs, 2).
            target (torch.Tensor): Ground truth targets (Batch, Max_Objs, 2).
            mask (torch.Tensor): Mask indicating valid objects (Batch, Max_Objs).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        expand_mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction="sum")

        # Normalize by the number of objects
        loss = loss / (mask.float().sum() + 1e-4)
        return loss


class CenterNetLoss(nn.Module):
    """
    Composite loss function for CenterNet.
    Combines Heatmap Loss (Modified Focal), Size Loss (L1), and Offset Loss (L1).
    """

    def __init__(self, wh_weight=0.1, reg_weight=1.0, hm_weight=1.0):
        super(CenterNetLoss, self).__init__()
        self.wh_weight = wh_weight
        self.reg_weight = reg_weight
        self.hm_weight = hm_weight

        self.hm_loss = ModifiedFocalLoss()
        self.wh_loss = RegL1Loss()
        self.reg_loss = RegL1Loss()

    def forward(self, outputs, batch):
        """
        Args:
            outputs (tuple): (hm_pred, wh_pred, reg_pred) from the model.
            batch (dict): Batch dictionary containing ground truths.

        Returns:
            tuple: (total_loss, loss_hm, loss_wh, loss_reg)
        """
        hm_pred, wh_pred, reg_pred = outputs

        # Ensure targets are on the correct device
        device = hm_pred.device
        hm_true = batch["hm"].to(device)
        wh_true = batch["wh"].to(device)
        reg_true = batch["reg"].to(device)
        ind_true = batch["ind"].to(device)
        reg_mask = batch["reg_mask"].to(device)

        # 1. Heatmap Loss
        loss_hm = self.hm_loss(hm_pred, hm_true)

        # 2. Gather predictions at ground truth center locations
        # wh_pred: (B, 2, H, W) -> (B, K, 2)
        wh_pred_gathered = transpose_and_gather_feat(wh_pred, ind_true)
        # reg_pred: (B, 2, H, W) -> (B, K, 2)
        reg_pred_gathered = transpose_and_gather_feat(reg_pred, ind_true)

        # 3. Regression Losses
        loss_wh = self.wh_loss(wh_pred_gathered, wh_true, reg_mask)
        loss_reg = self.reg_loss(reg_pred_gathered, reg_true, reg_mask)

        # 4. Weighted Sum
        total_loss = (
            (self.hm_weight * loss_hm)
            + (self.wh_weight * loss_wh)
            + (self.reg_weight * loss_reg)
        )

        return total_loss, loss_hm, loss_wh, loss_reg
