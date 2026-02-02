import torch
import torch.nn as nn
import torch.nn.functional as F


class ModifiedFocalLoss(nn.Module):
    """
    Penalty-reduced Focal Loss for the objectness heatmap.
    Designed to handle the class imbalance between the few center points and the background.
    """

    def __init__(self, alpha=2, beta=4):
        super(ModifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Predicted heatmap (B, 1, H, W), passed through sigmoid.
            gt (torch.Tensor): Ground truth heatmap (B, 1, H, W), values in [0, 1].

        Returns:
            torch.Tensor: Scalar loss value.
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.beta)

        # Clamp predictions for numerical stability
        pred = torch.clamp(pred, 1e-12, 1 - 1e-12)

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


class MaskedCrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss applied only at positive object centers.
    """

    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred (torch.Tensor): Predictions (B, C, H, W).
            target (torch.Tensor): Class indices (B, H, W).
            mask (torch.Tensor): Mask indicating object centers (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        b, c, h, w = pred.shape

        # Identify valid indices (B, H, W)
        valid_inds = mask.squeeze(1) > 0

        if valid_inds.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        # Select valid predictions: (B, C, H, W) -> (B, H, W, C) -> (N_valid, C)
        # We use boolean indexing which avoids creating the huge reshaped tensor
        pred_valid = pred.permute(0, 2, 3, 1)[valid_inds]

        # Select valid targets: (B, H, W) -> (N_valid,)
        target_valid = target[valid_inds]

        loss = F.cross_entropy(pred_valid, target_valid)
        return loss


class RegL1Loss(nn.Module):
    """
    L1 Loss for regression targets (offsets and dimensions), applied only at object centers.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred (torch.Tensor): Predictions (B, 4, H, W).
            target (torch.Tensor): Ground truth (B, 4, H, W).
            mask (torch.Tensor): Mask indicating object centers (B, 1, H, W).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Apply mask
        # We use sum reduction and divide by number of positive samples
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")

        # Normalize by number of objects (mask sum).
        num_objs = mask.sum()
        loss = loss / (num_objs + 1e-4)

        return loss


class DKNLoss(nn.Module):
    """
    Composite loss function for the Decoupled Keypoint Network.
    Combines heatmap focal loss, classification cross-entropy, and regression L1 loss.
    """

    def __init__(self):
        super(DKNLoss, self).__init__()
        self.hm_loss = ModifiedFocalLoss()
        self.cls_loss = MaskedCrossEntropyLoss()
        self.reg_loss = RegL1Loss()

    def forward(self, outputs, batch):
        """
        Args:
            outputs (dict): Model outputs containing 'hm', 'cls', 'reg'.
            batch (dict): Batch data containing 'hm', 'cls_target', 'reg_target', 'reg_mask'.

        Returns:
            tuple: (total_loss, stats_dict)
        """
        hm_pred = outputs["hm"]
        cls_pred = outputs["cls"]
        reg_pred = outputs["reg"]

        hm_target = batch["hm"].to(hm_pred.device)
        cls_target = batch["cls_target"].to(cls_pred.device)
        reg_target = batch["reg_target"].to(reg_pred.device)
        reg_mask = batch["reg_mask"].to(reg_pred.device)

        loss_hm = self.hm_loss(hm_pred, hm_target)
        loss_cls = self.cls_loss(cls_pred, cls_target, reg_mask)
        loss_reg = self.reg_loss(reg_pred, reg_target, reg_mask)

        # Weighted sum (Using 1.0 for all components)
        total_loss = loss_hm + loss_cls + loss_reg

        loss_stats = {
            "loss_hm": loss_hm.item(),
            "loss_cls": loss_cls.item(),
            "loss_reg": loss_reg.item(),
            "loss_total": total_loss.item(),
        }

        return total_loss, loss_stats
