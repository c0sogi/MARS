import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


def _gather_feat(feat, ind, mask=None):
    """
    Gathers features from a specific set of indices in the flattened spatial dimension.

    Args:
        feat (torch.Tensor): Feature map of shape (B, C, H, W) or (B, H, W, C) pre-permuted?
                             Standard CenterNet flow: Input is (B, C, H, W).
                             We permute to (B, H, W, C) then flatten to (B, H*W, C).
        ind (torch.Tensor): Indices of shape (B, K) where K is max objects.
        mask (torch.Tensor): Mask of shape (B, K) indicating valid objects.

    Returns:
        torch.Tensor: Gathered features of shape (B, K, C) or flattened if mask is applied.
    """
    dim = feat.size(1)  # Channels are at dim 1 initially
    # Permute to (B, H, W, C)
    feat = feat.permute(0, 2, 3, 1)
    # Flatten to (B, H*W, C)
    feat = feat.contiguous().view(feat.size(0), -1, dim)

    # Expand indices to (B, K, C)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)

    # Gather
    feat = feat.gather(1, ind)

    if mask is not None:
        # Apply mask to filter valid objects
        # mask is (B, K), expand to (B, K, C)
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)

    return feat


def _transpose_and_gather_feat(feat, ind):
    """
    Helper to permute and gather without masking, returning (B, K, C).
    Used when we want to keep the batch structure for loss calculation before masking.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))
    feat = feat.gather(1, ind)
    return feat


class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss for Heatmap Regression (CornerNet/CenterNet variant).
    Penalizes easy negatives less and handles Gaussian-splatted targets.
    """

    def __init__(self, alpha=2, beta=4):
        super(ModifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Predicted logits of shape (B, C, H, W).
            target (torch.Tensor): Ground truth heatmap of shape (B, C, H, W). Values in [0, 1].
        """
        # Apply sigmoid to get probabilities
        pred = torch.sigmoid(pred)

        # Clamp for numerical stability
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)

        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        neg_weights = torch.pow(1 - target, self.beta)

        loss = 0

        # Positive loss: (1 - pred)^alpha * log(pred)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds

        # Negative loss: (1 - target)^beta * pred^alpha * log(1 - pred)
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
    L1 Loss for regression tasks (Size, Offset) at specific center indices.
    """

    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, pred, target, mask, ind):
        """
        Args:
            pred (torch.Tensor): Prediction map (B, C, H, W).
            target (torch.Tensor): Ground truth values (B, K, C).
            mask (torch.Tensor): Validity mask (B, K).
            ind (torch.Tensor): Indices of centers (B, K).
        """
        # Gather predictions at the specific indices
        # Output shape: (B, K, C)
        pred_gathered = _transpose_and_gather_feat(pred, ind)

        # Expand mask for channels: (B, K) -> (B, K, C)
        mask = mask.unsqueeze(2).expand_as(pred_gathered).float()

        # Compute L1 loss masked
        loss = F.l1_loss(pred_gathered * mask, target * mask, reduction="sum")

        # Normalize by number of objects + epsilon
        loss = loss / (mask.sum() + 1e-4)
        return loss


class GlobalBCELoss(nn.Module):
    """
    Binary Cross Entropy for the global 'No Finding' classification head.
    """

    def __init__(self):
        super(GlobalBCELoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): Logits (B, 1).
            target (torch.Tensor): Binary labels (B, 1).
        """
        return self.loss_fn(pred, target)


class CenterNetLoss(nn.Module):
    """
    Composite loss function for the Task-Aligned Spatially-Decoupled CenterNet.
    Combines Heatmap Focal Loss, Regression L1 Loss, and Global BCE Loss.
    """

    def __init__(self, hm_weight=1.0, wh_weight=0.1, off_weight=1.0, global_weight=1.0):
        super(CenterNetLoss, self).__init__()
        self.hm_weight = hm_weight
        self.wh_weight = wh_weight
        self.off_weight = off_weight
        self.global_weight = global_weight

        self.hm_loss = ModifiedFocalLoss()
        self.reg_loss = RegL1Loss()
        self.global_loss = GlobalBCELoss()

    def forward(self, outputs, batch):
        """
        Args:
            outputs (dict): Dictionary containing model predictions:
                - 'hm': (B, NumClasses, H, W)
                - 'wh': (B, 2, H, W)
                - 'reg': (B, 2, H, W)
                - 'global_cls': (B, 1)
            batch (dict): Dictionary containing ground truth:
                - 'hm': (B, NumClasses, H, W)
                - 'wh': (B, K, 2)
                - 'reg': (B, K, 2)
                - 'ind': (B, K)
                - 'reg_mask': (B, K)
                - 'global_label': (B, 1)

        Returns:
            torch.Tensor: Weighted sum of losses.
            dict: Dictionary of individual loss components for logging.
        """

        # 1. Heatmap Loss
        hm_loss = self.hm_loss(outputs["hm"], batch["hm"])

        # 2. Width/Height Regression Loss
        wh_loss = self.reg_loss(
            outputs["wh"], batch["wh"], batch["reg_mask"], batch["ind"]
        )

        # 3. Offset Regression Loss
        off_loss = self.reg_loss(
            outputs["reg"], batch["reg"], batch["reg_mask"], batch["ind"]
        )

        # 4. Global Classification Loss
        # Ensure target is float for BCE
        global_loss = self.global_loss(
            outputs["global_cls"], batch["global_label"].float()
        )

        # Weighted Sum
        total_loss = (
            self.hm_weight * hm_loss
            + self.wh_weight * wh_loss
            + self.off_weight * off_loss
            + self.global_weight * global_loss
        )

        loss_stats = {
            "loss": total_loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
            "global_loss": global_loss,
        }

        return total_loss, loss_stats
