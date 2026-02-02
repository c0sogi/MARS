import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# Lovasz-Softmax / Hinge Loss Helpers
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
    Binary Lovasz hinge loss on flattened inputs
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


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = 0
        batch_size = len(logits)
        for i in range(batch_size):
            l_flat, lab_flat = flatten_binary_scores(logits[i], labels[i], ignore)
            loss += lovasz_hinge_flat(l_flat, lab_flat)
        return loss / batch_size
    else:
        l_flat, lab_flat = flatten_binary_scores(logits, labels, ignore)
        return lovasz_hinge_flat(l_flat, lab_flat)


# -----------------------------------------------------------------------------
# Loss Modules
# -----------------------------------------------------------------------------


class LovaszLoss(nn.Module):
    """
    Lovasz-Hinge loss for binary segmentation.
    Optimizes the Jaccard index (IoU) directly.
    """

    def __init__(self, per_image=True, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W) or (B, H, W)
            targets: (B, 1, H, W) or (B, H, W)
        """
        # Squeeze channel dim if present to match (B, H, W) expected by helper
        if logits.dim() > 3:
            logits = logits.squeeze(1)
        if targets.dim() > 3:
            targets = targets.squeeze(1)

        return lovasz_hinge(
            logits, targets, per_image=self.per_image, ignore=self.ignore
        )


class BCEDiceLoss(nn.Module):
    """
    Composite loss combining Binary Cross Entropy and Dice Loss.
    Used for stable initial convergence.
    """

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1, H, W)
            targets: (B, 1, H, W)
        """
        # 1. Binary Cross Entropy Loss
        bce_loss = self.bce_fn(logits, targets)

        # 2. Dice Loss
        logits_sigmoid = torch.sigmoid(logits)

        # Flatten batch for Dice calculation: (B, -1)
        batch_size = logits.size(0)
        logits_flat = logits_sigmoid.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        intersection = (logits_flat * targets_flat).sum(1)
        union = logits_flat.sum(1) + targets_flat.sum(1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
