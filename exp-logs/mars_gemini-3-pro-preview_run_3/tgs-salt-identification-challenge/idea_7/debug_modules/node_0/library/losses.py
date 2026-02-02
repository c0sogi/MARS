import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DiceLoss(nn.Module):
    """
    Dice Coefficient Loss for binary segmentation.
    Computes 1 - Dice.
    Expects logits as input (applies sigmoid internally).
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # preds: (B, 1, H, W) logits
        # targets: (B, 1, H, W) or (B, H, W) 0/1

        # Apply sigmoid to logits
        preds = torch.sigmoid(preds)

        # Flatten
        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross Entropy and Dice Loss.
    Supports Deep Supervision by averaging loss across all auxiliary outputs.
    """

    def __init__(self, bce_weight=0.5, smooth=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor of shape (B, 1, H, W) OR list/tuple of Tensors for Deep Supervision.
            targets: Tensor of shape (B, 1, H, W) or (B, H, W).
        """
        # Ensure targets have channel dim for BCE
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)

        targets = targets.float()

        # Handle Deep Supervision (List of outputs)
        if isinstance(preds, (list, tuple)):
            total_loss = 0
            for p in preds:
                bce_loss = self.bce(p, targets)
                dice_loss = self.dice(p, targets)
                total_loss += (bce_loss * self.bce_weight) + (
                    dice_loss * (1 - self.bce_weight)
                )
            return total_loss / len(preds)

        # Single Output
        else:
            bce_loss = self.bce(preds, targets)
            dice_loss = self.dice(preds, targets)
            return (bce_loss * self.bce_weight) + (dice_loss * (1 - self.bce_weight))


# -----------------------------------------------------------------------------
# Lovasz-Hinge Loss Implementation
# Adapted from: https://github.com/bermanmaxim/LovaszSoftmax
# -----------------------------------------------------------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0

    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


class LovaszHingeLoss(nn.Module):
    """
    Lovasz-Hinge Loss for optimizing Jaccard Index (IoU).
    Designed for the fine-tuning phase.
    If Deep Supervision inputs are provided (list), it only calculates loss on the FINAL output.
    """

    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, preds, targets):
        """
        Args:
            preds: Tensor of shape (B, 1, H, W) OR list/tuple of Tensors.
            targets: Tensor of shape (B, 1, H, W) or (B, H, W).
        """
        # Handle Deep Supervision: Select only the final output (index -1)
        if isinstance(preds, (list, tuple)):
            preds = preds[-1]

        # Ensure targets have channel dim removed for flattening logic consistency
        if targets.ndim == 4:
            targets = targets.squeeze(1)

        # Flatten
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        return lovasz_hinge_flat(preds_flat, targets_flat)
