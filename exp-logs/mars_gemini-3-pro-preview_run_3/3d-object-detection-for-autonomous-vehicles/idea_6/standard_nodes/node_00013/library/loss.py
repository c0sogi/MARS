import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def gather_feat(feat, ind, mask=None):
    """
    Gather feature at specified indices.
    Args:
        feat: (B, C, H, W) - Dense feature map
        ind: (B, K) - Indices in [0, H*W)
        mask: (B, K) - Binary mask (optional)
    Returns:
        feat: (B, K, C) - Gathered features
    """
    dim = feat.size(2) * feat.size(3)
    # ind: (B, K) -> (B, K, C)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(1))

    # feat: (B, C, H, W) -> (B, H, W, C) -> (B, H*W, C)
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(1))

    # Gather
    feat = feat.gather(1, ind)

    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, feat.size(2))

    return feat


class GaussianFocalLoss(nn.Module):
    """
    Gaussian Focal Loss for CenterPoint Heatmap.
    """

    def __init__(self, alpha=2, beta=4):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - Logits from the model
            target: (B, C, H, W) - Ground Truth Gaussian Heatmap [0, 1]
        """
        # Apply sigmoid to logits to get probabilities
        pred = torch.sigmoid(pred)
        # Clamp for numerical stability
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        pos_inds = target.eq(1)
        neg_inds = target.lt(1)

        # Weight for negative samples based on proximity to GT (Gaussian tail)
        neg_weights = torch.pow(1 - target, self.beta)

        loss = 0

        # Loss for positive samples (peaks)
        # (1 - pred)^alpha * log(pred)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds

        # Loss for negative samples (background + gaussian tails)
        # (1 - target)^beta * pred^alpha * log(1 - pred)
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
    L1 Loss applied only at Ground Truth indices (Masked).
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask):
        """
        Args:
            pred: (B, K, C) - Gathered predictions
            target: (B, K, C) - Regression targets
            mask: (B, K) - Validity mask (1 for object, 0 for padding)
        """
        assert pred.size() == target.size()

        loss = F.l1_loss(pred, target, reduction="none")

        # Apply mask
        mask_expanded = mask.unsqueeze(2).expand_as(loss)
        loss = loss * mask_expanded

        # Normalize by number of objects
        normalizer = mask.sum() + 1e-4
        loss = loss.sum() / normalizer

        return loss


class DetectionLoss(nn.Module):
    """
    Composite loss module for CenterPoint 3D Detection.
    """

    def __init__(self, config=None):
        super(DetectionLoss, self).__init__()
        self.config = config
        self.heatmap_loss = GaussianFocalLoss()
        self.reg_loss = RegL1Loss()

        # Default weights
        self.loss_weights = {
            "heatmap": 1.0,
            "offset": 1.0,
            "height": 1.0,
            "dim": 1.0,
            "rot": 1.0,
        }

    def forward(self, preds_dict, targets_dict):
        """
        Calculate total loss.

        Args:
            preds_dict: Dictionary of model outputs
                - heatmap: (B, C, H, W)
                - offset: (B, 2, H, W)
                - height: (B, 1, H, W)
                - dim: (B, 3, H, W)
                - rot: (B, 2, H, W)
            targets_dict: Dictionary of ground truth targets
                - heatmap: (B, C, H, W)
                - inds: (B, K) - Indices of objects
                - mask: (B, K) - Valid object mask
                - offset: (B, K, 2)
                - height: (B, K, 1)
                - dim: (B, K, 3)
                - rot: (B, K, 2)

        Returns:
            total_loss: Scalar tensor
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. Heatmap Loss
        if "heatmap" in preds_dict and "heatmap" in targets_dict:
            hm_pred = preds_dict["heatmap"]
            hm_target = targets_dict["heatmap"]
            hm_loss = self.heatmap_loss(hm_pred, hm_target)

            loss_dict["loss_heatmap"] = hm_loss
            total_loss += self.loss_weights["heatmap"] * hm_loss

        # 2. Regression Losses
        # We gather features from the dense prediction maps using the GT indices
        inds = targets_dict.get("inds")
        mask = targets_dict.get("mask")

        if inds is not None and mask is not None:
            # List of regression heads to process
            reg_heads = ["offset", "height", "dim", "rot"]

            for head in reg_heads:
                if head in preds_dict and head in targets_dict:
                    # Dense prediction: (B, C, H, W)
                    pred_map = preds_dict[head]
                    # Target: (B, K, C)
                    target_val = targets_dict[head]

                    # Gather: (B, K, C)
                    pred_gathered = gather_feat(pred_map, inds)

                    # Compute Masked L1 Loss
                    l_loss = self.reg_loss(pred_gathered, target_val, mask)

                    loss_dict[f"loss_{head}"] = l_loss
                    total_loss += self.loss_weights.get(head, 1.0) * l_loss

        loss_dict["total_loss"] = total_loss
        return total_loss, loss_dict
