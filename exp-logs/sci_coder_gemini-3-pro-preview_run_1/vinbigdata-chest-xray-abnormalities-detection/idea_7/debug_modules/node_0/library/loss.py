import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CenterNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_heatmap = Config.LOSS_WEIGHT_HEATMAP
        self.w_size = Config.LOSS_WEIGHT_SIZE
        self.w_offset = Config.LOSS_WEIGHT_OFFSET
        self.w_global = Config.LOSS_WEIGHT_GLOBAL

        # Auxiliary Global Head Loss
        self.global_loss_fn = nn.BCEWithLogitsLoss()

    def modified_focal_loss(self, pred_logits, gt_heatmap):
        """
        Modified Focal Loss for Heatmap Regression.
        Penalizes easy negatives less and focuses on hard examples.

        Args:
            pred_logits: (B, C, H, W) - Raw logits from the model
            gt_heatmap: (B, C, H, W) - Gaussian smoothed targets (0-1)
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred_logits)

        # Clamp for numerical stability
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        # Identify positive (center) and negative samples
        # Note: gt_heatmap is Gaussian smoothed, so exact 1s are centers
        pos_inds = gt_heatmap.eq(1).float()
        neg_inds = gt_heatmap.lt(1).float()

        # Weighting for negative samples (penalty reduced near the center)
        neg_weights = torch.pow(1 - gt_heatmap, 4)

        # Standard Focal Loss parameters
        alpha = 2.0

        # Loss calculation
        # Log(p) for positives
        pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds

        # Log(1-p) for negatives, weighted by distance from center
        neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

        # Normalize by number of objects
        num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss

    def reg_l1_loss(self, pred, target, mask):
        """
        L1 Loss masked by object presence.

        Args:
            pred: (B, 2, H, W)
            target: (B, 2, H, W)
            mask: (B, H, W) - 1 at object center, 0 otherwise
        """
        # Expand mask to match channel dimension (B, 1, H, W) -> (B, 2, H, W)
        mask = mask.unsqueeze(1).expand_as(pred)

        # Calculate element-wise L1 loss
        loss = F.l1_loss(pred, target, reduction="none")

        # Apply mask
        loss = loss * mask

        # Normalize by number of objects (sum of mask)
        # Add epsilon to prevent division by zero
        normalizer = mask.sum() + 1e-4

        loss = loss.sum() / normalizer
        return loss

    def forward(self, outputs, targets):
        """
        Calculate total loss.

        Args:
            outputs: Dict containing 'heatmap', 'wh', 'offset', 'global_logits'
            targets: Dict containing 'heatmap', 'wh', 'offset', 'reg_mask', 'global_label'
        """

        # 1. Heatmap Loss
        hm_loss = self.modified_focal_loss(outputs["heatmap"], targets["heatmap"])

        # 2. Regression Losses (Size and Offset)
        # Only calculated where objects exist (reg_mask)
        wh_loss = self.reg_l1_loss(outputs["wh"], targets["wh"], targets["reg_mask"])
        off_loss = self.reg_l1_loss(
            outputs["offset"], targets["offset"], targets["reg_mask"]
        )

        # 3. Global Classification Loss (Auxiliary)
        # Ensure target shape matches logits (B, 1)
        global_target = targets["global_label"].view(-1, 1)
        global_loss = self.global_loss_fn(outputs["global_logits"], global_target)

        # Weighted Sum
        total_loss = (
            self.w_heatmap * hm_loss
            + self.w_size * wh_loss
            + self.w_offset * off_loss
            + self.w_global * global_loss
        )

        loss_stats = {
            "loss": total_loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
            "global_loss": global_loss,
        }

        return total_loss, loss_stats
