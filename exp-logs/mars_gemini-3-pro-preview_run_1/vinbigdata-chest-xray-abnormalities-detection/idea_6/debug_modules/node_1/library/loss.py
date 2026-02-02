import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss for CenterNet heatmaps.
    Penalizes easy negatives less and focuses on hard examples.
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        Args:
            pred (torch.Tensor): Predicted heatmap (B, C, H, W), range [0, 1].
            gt (torch.Tensor): Ground truth heatmap (B, C, H, W), range [0, 1].
        """
        # Clamp for numerical stability
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # Weight for negative examples based on distance from center (gt value)
        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0

        # Positive loss: -log(pred) * (1-pred)^alpha
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds

        # Negative loss: -log(1-pred) * pred^alpha * (1-gt)^beta
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


class MaskedL1Loss(nn.Module):
    """
    L1 Loss applied only at positive locations (mask == 1).
    Used for Size and Offset regression.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred (torch.Tensor): (B, 2, H, W)
            target (torch.Tensor): (B, 2, H, W)
            mask (torch.Tensor): (B, 1, H, W)
        """
        # Expand mask to match channel dim
        mask = mask.expand_as(pred)

        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")

        # Normalize by number of objects (sum of mask / channels)
        # Add epsilon to avoid division by zero
        num_objs = mask.sum() / 2 + 1e-4
        loss = loss / num_objs

        return loss


class ThoracicLoss(nn.Module):
    """
    Composite loss function for Thoracic Disease Detection.
    Combines:
    1. Modified Focal Loss (Heatmap)
    2. Masked L1 Loss (Size)
    3. Masked L1 Loss (Offset)
    4. Binary Cross Entropy (Global Classification)
    """

    def __init__(self, weight_hm=1.0, weight_wh=0.1, weight_off=1.0, weight_global=1.0):
        super().__init__()
        self.weight_hm = weight_hm
        self.weight_wh = weight_wh
        self.weight_off = weight_off
        self.weight_global = weight_global

        self.crit_hm = ModifiedFocalLoss()
        self.crit_reg = MaskedL1Loss()
        self.crit_global = nn.BCELoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict):
                'heatmap': (B, C, H, W)
                'size': (B, 2, H, W)
                'offset': (B, 2, H, W)
                'global_prob': (B, 1)
            targets (dict):
                'heatmap': (B, C, H, W)
                'size': (B, 2, H, W)
                'offset': (B, 2, H, W)
                'mask': (B, 1, H, W)
                'global_label': (B, 1)

        Returns:
            total_loss (torch.Tensor): Scalar loss for backprop.
            stats (dict): Dictionary of individual loss components for logging.
        """
        # 1. Heatmap Loss
        hm_loss = self.crit_hm(outputs["heatmap"], targets["heatmap"])

        # 2. Regression Losses
        # Only calculate if there are objects in the batch (mask sum > 0)
        # The MaskedL1Loss handles normalization internally
        wh_loss = self.crit_reg(outputs["size"], targets["size"], targets["mask"])
        off_loss = self.crit_reg(outputs["offset"], targets["offset"], targets["mask"])

        # 3. Global Classification Loss
        # Ensure targets are float for BCELoss
        global_loss = self.crit_global(
            outputs["global_prob"], targets["global_label"].float()
        )

        # 4. Weighted Sum
        total_loss = (
            self.weight_hm * hm_loss
            + self.weight_wh * wh_loss
            + self.weight_off * off_loss
            + self.weight_global * global_loss
        )

        loss_stats = {
            "loss": total_loss.item(),
            "hm_loss": hm_loss.item(),
            "wh_loss": wh_loss.item(),
            "off_loss": off_loss.item(),
            "global_loss": global_loss.item(),
        }

        return total_loss, loss_stats
