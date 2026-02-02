import torch
import torch.nn as nn
import torch.nn.functional as F
import library.config as config


def modified_focal_loss(pred, target):
    """
    Modified Focal Loss for Heatmap Regression (CenterNet style).

    Args:
        pred (torch.Tensor): Predicted heatmap (after sigmoid), shape (B, C, H, W).
        target (torch.Tensor): Ground truth heatmap with Gaussian peaks, shape (B, C, H, W).

    Returns:
        torch.Tensor: Scalar loss.
    """
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, 4)

    loss = 0

    # Clamp predictions to avoid log(0)
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

    # Positive loss: -log(pred) * (1 - pred)^2
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds

    # Negative loss: -log(1 - pred) * pred^2 * (1 - target)^4
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos

    return loss


def reg_l1_loss(pred, target, mask):
    """
    L1 Regression Loss applied only at object centers.

    Args:
        pred (torch.Tensor): Prediction map (B, C, H, W).
        target (torch.Tensor): Target map (B, C, H, W).
        mask (torch.Tensor): Mask of object centers (B, 1, H, W).

    Returns:
        torch.Tensor: Scalar loss.
    """
    # Expand mask to match channel dimension
    expand_mask = mask.expand_as(pred)

    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction="sum")

    # Normalize by number of objects
    mask_sum = mask.sum()
    loss = loss / (mask_sum + 1e-4)

    return loss


class BEVLoss(nn.Module):
    """
    Composite loss function for Rasterized BEV Detection.
    Combines Focal Loss for heatmaps and L1 Loss for regression targets.
    """

    def __init__(self):
        super(BEVLoss, self).__init__()
        self.hm_weight = config.HM_WEIGHT
        self.wh_weight = config.WH_WEIGHT
        self.off_weight = config.OFF_WEIGHT
        self.z_weight = config.Z_WEIGHT
        self.rot_weight = config.ROT_WEIGHT

    def forward(self, preds, targets):
        """
        Calculate the total loss.

        Args:
            preds (tuple): (hm_logits, reg_map)
                - hm_logits: (B, NumClasses, H, W)
                - reg_map: (B, 8, H, W)
            targets (dict):
                - hm: (B, NumClasses, H, W)
                - reg: (B, 8, H, W)
                - reg_mask: (B, 1, H, W)

        Returns:
            loss (torch.Tensor): Weighted sum of losses.
            stats (dict): Dictionary of individual loss components for logging.
        """
        hm_pred_logits, reg_pred = preds
        hm_target = targets["hm"]
        reg_target = targets["reg"]
        reg_mask = targets["reg_mask"]

        # 1. Heatmap Loss
        # Apply sigmoid to logits
        hm_pred = torch.sigmoid(hm_pred_logits)
        hm_loss = modified_focal_loss(hm_pred, hm_target)

        # 2. Regression Losses
        # Regression channels:
        # 0: off_x, 1: off_y
        # 2: z
        # 3: log(w), 4: log(l), 5: log(h)
        # 6: sin, 7: cos

        # Offset Loss (Channels 0, 1)
        off_loss = reg_l1_loss(
            reg_pred[:, 0:2, :, :], reg_target[:, 0:2, :, :], reg_mask
        )

        # Z Coordinate Loss (Channel 2)
        z_loss = reg_l1_loss(reg_pred[:, 2:3, :, :], reg_target[:, 2:3, :, :], reg_mask)

        # Dimension Loss (Channels 3, 4, 5)
        wh_loss = reg_l1_loss(
            reg_pred[:, 3:6, :, :], reg_target[:, 3:6, :, :], reg_mask
        )

        # Rotation Loss (Channels 6, 7)
        rot_loss = reg_l1_loss(
            reg_pred[:, 6:8, :, :], reg_target[:, 6:8, :, :], reg_mask
        )

        # 3. Total Loss
        loss = (
            self.hm_weight * hm_loss
            + self.off_weight * off_loss
            + self.z_weight * z_loss
            + self.wh_weight * wh_loss
            + self.rot_weight * rot_loss
        )

        stats = {
            "loss": loss.item(),
            "hm_loss": hm_loss.item(),
            "off_loss": off_loss.item(),
            "z_loss": z_loss.item(),
            "wh_loss": wh_loss.item(),
            "rot_loss": rot_loss.item(),
        }

        return loss, stats
