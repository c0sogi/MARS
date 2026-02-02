import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class CenterNetLoss(nn.Module):
    """
    Multi-task loss for Spatially-Aware CenterNet.

    Components:
    1. Modified Focal Loss for Heatmap (Classification)
    2. L1 Loss for Size (Regression, masked by object presence)
    3. L1 Loss for Offset (Regression, masked by object presence)
    4. BCE Loss for Global Classification (Finding vs No Finding)
    """

    def __init__(self):
        super(CenterNetLoss, self).__init__()
        self.lambda_hm = Config.LAMBDA_HEATMAP
        self.lambda_wh = Config.LAMBDA_SIZE
        self.lambda_reg = Config.LAMBDA_OFFSET
        self.lambda_global = Config.LAMBDA_GLOBAL

        # Binary Cross Entropy for the global classification head
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs, batch):
        """
        Calculates the weighted sum of all loss components.

        Args:
            outputs (dict): Dictionary containing model predictions:
                - 'hm': Heatmap logits [B, C, H, W]
                - 'wh': Size predictions [B, 2, H, W]
                - 'reg': Offset predictions [B, 2, H, W]
                - 'global': Global classification logits [B, 1]
            batch (dict): Dictionary containing ground truth targets:
                - 'target_heatmap': [B, C, H, W]
                - 'target_size': [B, 2, H, W]
                - 'target_offset': [B, 2, H, W]
                - 'target_mask': [B, H, W]
                - 'global_label': [B, 1]

        Returns:
            tuple: (total_loss, stats_dict)
        """
        pred_hm = outputs["hm"]
        pred_wh = outputs["wh"]
        pred_reg = outputs["reg"]
        pred_global = outputs["global"]

        # Move targets to the same device as predictions
        target_hm = batch["target_heatmap"].to(pred_hm.device)
        target_wh = batch["target_size"].to(pred_wh.device)
        target_reg = batch["target_offset"].to(pred_reg.device)
        target_mask = batch["target_mask"].to(pred_hm.device)
        target_global = batch["global_label"].to(pred_global.device)

        # ---------------------------------------------------------------------
        # 1. Heatmap Loss (Modified Focal Loss)
        # ---------------------------------------------------------------------
        # Apply sigmoid to logits to get probabilities in [0, 1]
        pred_hm_sigmoid = torch.sigmoid(pred_hm)
        hm_loss = self._modified_focal_loss(pred_hm_sigmoid, target_hm)

        # ---------------------------------------------------------------------
        # 2. Regression Losses (L1 Loss)
        # ---------------------------------------------------------------------
        # We only calculate regression loss where an object exists (target_mask == 1)
        # Expand mask to match channel dimensions of predictions (B, 2, H, W)
        mask = target_mask.unsqueeze(1).expand_as(pred_wh)

        # Count number of objects to normalize the loss
        # Add epsilon to prevent division by zero if batch has no objects
        num_objs = mask.sum() + 1e-4

        # Size Loss
        wh_loss = (
            F.l1_loss(pred_wh * mask, target_wh * mask, reduction="sum") / num_objs
        )

        # Offset Loss
        reg_loss = (
            F.l1_loss(pred_reg * mask, target_reg * mask, reduction="sum") / num_objs
        )

        # ---------------------------------------------------------------------
        # 3. Global Classification Loss (BCE)
        # ---------------------------------------------------------------------
        global_loss = self.bce(pred_global, target_global)

        # ---------------------------------------------------------------------
        # Total Loss
        # ---------------------------------------------------------------------
        loss = (
            self.lambda_hm * hm_loss
            + self.lambda_wh * wh_loss
            + self.lambda_reg * reg_loss
            + self.lambda_global * global_loss
        )

        stats = {
            "loss": loss.item(),
            "hm_loss": hm_loss.item(),
            "wh_loss": wh_loss.item(),
            "reg_loss": reg_loss.item(),
            "global_loss": global_loss.item(),
        }

        return loss, stats

    def _modified_focal_loss(self, pred, gt):
        """
        Modified focal loss from CornerNet/CenterNet papers.
        Penalizes easy negatives less and focuses on hard examples.

        Args:
            pred (torch.Tensor): Predicted heatmap probabilities [0, 1].
            gt (torch.Tensor): Ground truth heatmap [0, 1].

        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Identify positive (center) and negative pixels
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # Weight for negative samples:
        # Points closer to the ground truth center (higher gt value) get less penalty
        # This allows the model to predict high values near the center without harsh penalty
        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        # Positive loss: Standard Focal Loss term for p=1
        # -log(p) * (1-p)^alpha
        pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 2) * pos_inds

        # Negative loss: Standard Focal Loss term for p=0, weighted by distance
        # -log(1-p) * p^alpha * weight
        neg_loss = (
            torch.log(1 - pred + 1e-12) * torch.pow(pred, 2) * neg_weights * neg_inds
        )

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss
