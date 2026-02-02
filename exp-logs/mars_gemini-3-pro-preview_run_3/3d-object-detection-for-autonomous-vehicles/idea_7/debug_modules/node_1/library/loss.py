import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import LOSS_WEIGHTS
from library.utils import _transpose_and_gather_feat


class FastFocalLoss(nn.Module):
    """
    Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory.
    Arguments:
        pred (batch x c x h x w)
        gt (batch x c x h x w)
    """

    def __init__(self):
        super(FastFocalLoss, self).__init__()

    def forward(self, out, target, ind=None, mask=None, cat=None):
        """
        Modified focal loss. Exactly the same as CornerNet.
        Arguments:
            out (batch x c x h x w)
            target (batch x c x h x w)
        """
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()

        neg_weights = torch.pow(1 - target, 4)

        loss = 0

        pred = torch.clamp(torch.sigmoid(out), min=1e-4, max=1 - 1e-4)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


class RegLoss(nn.Module):
    """
    L1 loss for regression tasks (offset, height, dim, rot).
    Only calculates loss at ground truth center locations.
    """

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        """
        Args:
            output: (B, C, H, W) Dense prediction map
            mask: (B, K) Mask indicating valid objects (1) vs padding (0)
            ind: (B, K) Indices of ground truth centers in flattened spatial array
            target: (B, K, C) Ground truth values at indices
        """
        # Transpose and gather features at specific indices
        # pred: (B, K, C)
        pred = _transpose_and_gather_feat(output, ind)

        # Calculate L1 loss
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")

        # Normalize by number of objects
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CenterLoss(nn.Module):
    """
    Composite loss module for Center-based 3D Object Detection.
    Combines Heatmap Focal Loss and Regression L1 Losses.
    """

    def __init__(self):
        super(CenterLoss, self).__init__()
        self.crit = FastFocalLoss()
        self.crit_reg = RegLoss()
        self.loss_weights = LOSS_WEIGHTS

    def forward(self, preds_dict, targets_dict):
        """
        Calculate total loss.

        Args:
            preds_dict: Dictionary containing model outputs:
                - 'heatmap': (B, C, H, W)
                - 'reg': (B, 2, H, W)
                - 'height': (B, 1, H, W)
                - 'dim': (B, 3, H, W)
                - 'rot': (B, 2, H, W)
            targets_dict: Dictionary containing ground truth targets:
                - 'heatmap': (B, C, H, W)
                - 'ind': (B, K)
                - 'mask': (B, K)
                - 'reg': (B, K, 2)
                - 'height': (B, K, 1)
                - 'dim': (B, K, 3)
                - 'rot': (B, K, 2)

        Returns:
            loss: Scalar tensor representing weighted sum of losses
            loss_stats: Dictionary of individual loss values (for logging)
        """
        loss = 0
        loss_stats = {}

        # 1. Heatmap Loss
        if "heatmap" in preds_dict and "heatmap" in targets_dict:
            hm_loss = self.crit(preds_dict["heatmap"], targets_dict["heatmap"])
            loss += self.loss_weights["heatmap"] * hm_loss
            loss_stats["loss_heatmap"] = hm_loss.item()

        # Common regression arguments
        ind = targets_dict["ind"]
        mask = targets_dict["mask"]

        # 2. Regression Losses
        # We iterate over the regression heads defined in config (implicitly via targets/preds keys)
        # Keys expected: 'reg', 'height', 'dim', 'rot'

        heads = ["reg", "height", "dim", "rot"]

        for head in heads:
            if head in preds_dict and head in targets_dict:
                # Calculate masked L1 loss
                reg_loss_val = self.crit_reg(
                    preds_dict[head], mask, ind, targets_dict[head]
                )

                # Weighted sum
                weight = self.loss_weights.get(head, 1.0)
                loss += weight * reg_loss_val

                # Log stats
                loss_stats[f"loss_{head}"] = reg_loss_val.item()

        return loss, loss_stats
