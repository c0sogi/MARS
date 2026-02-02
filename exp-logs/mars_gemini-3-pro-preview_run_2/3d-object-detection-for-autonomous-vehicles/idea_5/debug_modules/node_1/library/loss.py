import torch
import torch.nn as nn
import torch.nn.functional as F


class FastFocalLoss(nn.Module):
    """
    Penalty-Reduced Focal Loss for Heatmap Regression.
    Used in CenterNet and CenterPoint to handle the imbalance between
    sparse object centers and the dense background.
    """

    def __init__(self, alpha=2.0, beta=4.0):
        super(FastFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        Args:
            pred (Tensor): Predicted heatmap (B, C, H, W), values in [0, 1].
            gt (Tensor): Ground truth heatmap (B, C, H, W), values in [0, 1].
        Returns:
            Tensor: Scalar loss value.
        """
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # Weight negative samples by their distance to the center (1 - gt)^beta
        neg_weights = torch.pow(1 - gt, self.beta)

        # Clamp for numerical stability
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)

        # Positive loss: - (1 - pred)^alpha * log(pred)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds

        # Negative loss: - (1 - gt)^beta * pred^alpha * log(1 - pred)
        neg_loss = (
            torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        )

        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss


class RegLoss(nn.Module):
    """
    Masked L1 Loss for dense regression heads.
    Calculates L1 loss only at valid object center locations.
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, pred, gt, mask):
        """
        Args:
            pred (Tensor): Predicted regression map (B, C, H, W).
            gt (Tensor): Ground truth regression map (B, C, H, W).
            mask (Tensor): Boolean or Binary mask of object centers (B, 1, H, W).
        Returns:
            Tensor: Scalar loss value.
        """
        # Expand mask to match channel dimension if necessary
        if mask.shape[1] == 1 and pred.shape[1] > 1:
            mask = mask.expand_as(pred)

        loss = F.l1_loss(pred, gt, reduction="none")

        # Apply mask
        loss = loss * mask

        # Normalize by the number of positive samples (sum of mask)
        # We add a small epsilon to avoid division by zero
        normalizer = mask.sum()
        normalizer = torch.clamp(normalizer, min=1e-4)

        loss = loss.sum() / normalizer
        return loss


class CenterPointLoss(nn.Module):
    """
    Composite loss module for the CenterPoint architecture.
    Combines FastFocalLoss for the heatmap and RegLoss for 3D attributes.
    """

    def __init__(self):
        super(CenterPointLoss, self).__init__()
        self.focal_loss = FastFocalLoss()
        self.reg_loss = RegLoss()

        # Weights for different heads
        # These can be tuned based on the scale of the regression targets
        self.loss_weights = {
            "hm": 1.0,
            "center_z": 1.0,
            "dim": 2.0,  # Dimensions are critical for IoU
            "rot": 1.0,  # Orientation
            "reg": 1.0,  # Local offset
        }

    def forward(self, preds_dict, targets_dict):
        """
        Args:
            preds_dict (dict): Output from the model.
                               Keys: 'hm', 'center_z', 'dim', 'rot', 'reg'.
                               Values: Tensors of shape (B, C_head, H, W).
            targets_dict (dict): Ground truth targets.
                                 Keys matching preds_dict + 'mask_reg'.
                                 'mask_reg' is (B, 1, H, W) indicating object centers.
        Returns:
            dict: Dictionary containing 'loss' (total) and individual head losses.
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. Heatmap Loss
        if "hm" in preds_dict and "hm" in targets_dict:
            hm_pred = preds_dict["hm"]
            hm_target = targets_dict["hm"]

            # Apply sigmoid to convert logits to probabilities
            hm_pred = torch.sigmoid(hm_pred)

            hm_loss = self.focal_loss(hm_pred, hm_target)

            weight = self.loss_weights.get("hm", 1.0)
            loss_dict["loss_hm"] = hm_loss
            total_loss += hm_loss * weight

        # 2. Regression Losses
        # We use the same mask for all regression heads
        mask = targets_dict.get("mask_reg")

        if mask is not None:
            # Iterate over regression heads
            for head in ["center_z", "dim", "rot", "reg"]:
                if head in preds_dict and head in targets_dict:
                    pred = preds_dict[head]
                    target = targets_dict[head]

                    l = self.reg_loss(pred, target, mask)

                    weight = self.loss_weights.get(head, 1.0)
                    loss_dict[f"loss_{head}"] = l
                    total_loss += l * weight

        loss_dict["loss"] = total_loss
        return loss_dict
